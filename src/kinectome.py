import os
from src.data_utils import data_loader, groups
from src.preprocessing.preprocessing import all_preprocessing
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.dist_dependence_measures import distance_correlation
from tqdm import tqdm
from src.data_utils.plotting import visualise_kinectome
from pathlib import Path
from scipy import interpolate

def find_full_leftRight_cycles(events: pd.DataFrame, data: pd.DataFrame):
    """Build overlapping gait cycle windows from pre-detected gait events.

    Each cycle spans from a right initial contact (IC) to the second subsequent
    left IC, giving one complete left and one complete right gait cycle per
    window.  Consecutive windows overlap because the algorithm advances one
    right IC at a time.

    Gait events are detected once on the raw data (before preprocessing) in
    ``calculate_all_kinectomes``, then passed in here to avoid re-running
    detection on data that no longer has the cone marker columns.

    Parameters
    ----------
    events : pd.DataFrame
        Events table as returned by ``data_loader.detect_gait_events_from_data``
        or the old ``data_loader.load_events``.  Must have columns ``onset``
        and ``event_type`` with values ``'start'``, ``'stop'``,
        ``'initial_contact_left'``, ``'initial_contact_right'``.
        Onset values are in full-signal (untrimmed) frame space.
    data : pd.DataFrame
        Preprocessed (trimmed) motion capture data.  Used only to check
        that cycle end frames don't exceed the data length.

    Returns
    -------
    gait_cycles : list[tuple[int, int]]
        List of ``(start_frame, end_frame)`` pairs in 0-based trimmed-data
        frame space.
    start_onset : int
        Walkway gate start frame in the original untrimmed signal.  Used
        downstream to label saved kinectome files.
    """
    start_onset = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
    stop_onset  = int(events.loc[events['event_type'] == 'stop',  'onset'].values[0])

    # Shift IC indices to trimmed-data frame space (0-based from start_onset)
    events = events.copy()
    events['onset'] = events['onset'] - start_onset

    valid_events = events[
        (events['onset'] >= 0) & (events['onset'] <= (stop_onset - start_onset))
    ]

    icl_indices = valid_events[valid_events['event_type'] == 'initial_contact_left']['onset'].values
    icr_indices = valid_events[valid_events['event_type'] == 'initial_contact_right']['onset'].values

    gait_cycles = []
    i, j = 0, 0

    while i < len(icr_indices) - 1 and j < len(icl_indices) - 1:
        start_cycle = icr_indices[i]
        end_cycle = None

        for j in range(len(icl_indices)):
            if icl_indices[j] > start_cycle:
                if j + 1 < len(icl_indices):
                    end_cycle = icl_indices[j + 1]
                    break

        if end_cycle and start_cycle >= 0 and end_cycle <= len(data):
            gait_cycles.append((int(start_cycle), int(end_cycle)))

        i += 1

    return gait_cycles, start_onset

def segment_data(data: pd.DataFrame, cycle_indices: tuple):
    """
    Extracts a segment of motion tracking data corresponding to a specific gait cycle.

    Parameters:
    - data (pd.DataFrame): The motion tracking data.
    - cycle_indices (tuple): A tuple (start_index, end_index) defining the range of the gait cycle.

    Returns:
    - cycle_data (pd.DataFrame): A subset of the data corresponding to the given gait cycle.
    """

    cycle_data = data[cycle_indices[0]:cycle_indices[1]]

    return cycle_data

def timelag_cross_correlation_matrix(data: pd.DataFrame, marker_list: list):
    """
    Computes the time-lag cross-correlation matrix for given markers.

    Parameters:
    - data (pd.DataFrame): Motion tracking data.
    - marker_list (list): List of marker names.

    Returns:
    - corr_matrix (np.ndarray): Maximum cross-correlation values.
    - lag_matrix (np.ndarray): Corresponding time lags. 
        A positive lag value at lag_matrix[i, j] means signal j leads signal i by that percentage of the gait cycle. 
        The peak correlation is found when signal j is shifted backward in time (or signal i is shifted forward) by that amount.

    """
    n_markers = len(marker_list)
    corr_matrix = np.zeros((n_markers, n_markers))
    lag_matrix = np.zeros((n_markers, n_markers))

    # determine the length of the gait cycle
    gait_cycle_frames = len(data)

    for i in range(n_markers):
        for j in range(i + 1, n_markers):  # Compute only upper triangle (symmetric)
            sig1, sig2 = data[marker_list[i]].values, data[marker_list[j]].values

            # Compute normalized cross-correlation
            corr = np.correlate(sig1 - sig1.mean(), sig2 - sig2.mean(), mode='full')
            corr /= np.sqrt(np.sum((sig1 - sig1.mean())**2) * np.sum((sig2 - sig2.mean())**2))

            # Get max correlation and corresponding lag
            lags = np.arange(-len(sig1) + 1, len(sig1))
            max_idx = np.argmax(np.abs(corr))
            
            corr_matrix[i, j] = corr[max_idx]

            # Convert lag to percentage of gait cycle (with 2 decimal places)
            lag_percentage = (lags[max_idx] / gait_cycle_frames) * 100
            lag_matrix[i, j] = round(lag_percentage, 0)
            
            # Mirror results for symmetric matrix
            corr_matrix[j, i] = corr[max_idx]
            lag_matrix[j, i] = round(-lag_percentage, 0) # Negate for symmetry
    
    return corr_matrix, lag_matrix

def distance_correlation_matrix(data: pd.DataFrame, markers_list: list):
    """
    Computes the distance correlation matrix for marker positions in x, y, and z coordinates across gait cycles.

    Parameters:
    - data (pd.DataFrame): The motion tracking data.
    - marker_list (list): List of marker names.
    Returns:
    - distance_correlation_matrix (np.ndarray): The distance correlation matrix.
    """
    
    dcor = np.array([[distance_correlation(data[m1], data[m2]) for m2 in markers_list] for m1 in markers_list])

    return dcor

def calculate_kinectome(data: pd.DataFrame, events: pd.DataFrame,
                        sub_id: str, task_name: str, run: str, tracksys: str,
                        kinematics: str, base_path: str, result_base_path: str,
                        marker_list: list, full_kinectomes: bool,
                        correlation_method: str, linux: bool = False,
                        ):
    """Compute and save kinectome matrices for all gait cycles of one subject/task/run.

    Pearson, distance, or cross-correlation matrices are computed per gait cycle
    and per movement direction (AP, ML, V), then saved as ``.npy`` files under
    ``BASE_PATH/derived_data/sub-<id>/kinectomes/``.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed motion capture data (trimmed, gap-filled, rotated,
        differentiated).
    events : pd.DataFrame
        Gait events table with columns ``onset`` and ``event_type``.
        Detected once from the raw data (before preprocessing) in
        ``calculate_all_kinectomes`` and passed in to avoid re-running
        detection on data that no longer contains cone marker columns.
    sub_id : str
        Subject identifier (e.g. ``'pp065'``).
    task_name : str
        Task name (e.g. ``'walkStroop'``).
    run : str or None
        Run condition (``'on'``, ``'off'``, or ``None`` for controls).
    tracksys : str
        Tracking system (e.g. ``'omc'``).
    kinematics : str
        Kinematic signal used: ``'pos'``, ``'vel'``, or ``'acc'``.
    base_path : str or Path
        Root data directory.  Kinectomes are saved under
        ``base_path/derived_data/sub-<id>/kinectomes/``.
    result_base_path : str or Path
        Root results directory (used for visualisation outputs).
    marker_list : list[str]
        Ordered list of marker names to include in the kinectome.
    full_kinectomes : bool
        If True, compute one combined kinectome (all directions).
        If False, compute separate AP, ML, V kinectomes.
    correlation_method : str
        ``'pears'``, ``'cross'``, or ``'dcor'``.
    linux : bool, optional
        Path separator flag (legacy, kept for compatibility).
    interpol : bool, optional
        If True, interpolate each gait cycle to 500 frames before computing.

    Returns
    -------
    None
        Kinectome arrays are saved to disk as ``.npy`` files.
    """

    gait_cycles, start_onset = find_full_leftRight_cycles(events, data)
    cycles_iterator = tqdm(gait_cycles, desc=f"---Subject: {sub_id}, Task: {task_name}---")
    for i, cycles in enumerate(cycles_iterator): #range(len(gait_cycles)):
        cycle_indices = gait_cycles[i]

        gait_cycle_data = segment_data(data, cycle_indices)

        # always interpolate the data before calculating kinectomes
        # Assuming your data is in a DataFrame called 'df'
        original_length = len(gait_cycle_data)
        target_length = 500

        # Create original and target time indices
        original_indices = np.arange(original_length)
        target_indices = np.linspace(0, original_length-1, target_length)

        # Interpolate each column
        interpolated_data = {}
        for column in gait_cycle_data.columns:
            f = interpolate.interp1d(original_indices, gait_cycle_data[column], kind='linear')
            interpolated_data[column] = f(target_indices)

        # Create new DataFrame (once, after all columns are interpolated)
        gait_cycle_data = pd.DataFrame(interpolated_data)

        # Extract marker names that have ALL THREE axis columns present.
        # A marker is only usable if x, y and z all exist for this cycle.
        K = kinematics.upper()
        present_markers = [
            m for m in marker_list
            if all(f"{m}_{K}_{ax}" in gait_cycle_data.columns for ax in ("x", "y", "z"))
        ]

        # Guarantee whole-body kinectomes: if ANY requested marker is missing or
        # incomplete, skip this cycle entirely rather than save a reduced matrix.
        missing_markers = [m for m in marker_list if m not in present_markers]
        if missing_markers:
            print(f"  Missing/incomplete markers for sub-{sub_id}, task-{task_name}: "
                  f"{missing_markers} — skipping this cycle to keep kinectomes full-size.")
            continue

        # Reorder columns based on MARKER_LIST (all markers guaranteed present here).
        ordered_columns = []
        for marker in marker_list:
            ordered_columns.extend([f"{marker}_{K}_x",
                                    f"{marker}_{K}_y",
                                    f"{marker}_{K}_z"])

        # Subset and reorder dataframe
        gait_cycle_data = gait_cycle_data[ordered_columns]

        
        # compute correlations for all coordinates (x AND y AND z)        
        if full_kinectomes: 
            num_markers = len(marker_list)
            all_markers = list(gait_cycle_data.columns)
            correlation_matrix_full = np.zeros((num_markers*3, num_markers*3))
            timelag_matrix_full = np.zeros((num_markers*3, num_markers*3))
        
            if correlation_method == 'dcor':
                correlation_matrix_full = distance_correlation_matrix(gait_cycle_data[all_markers], all_markers)
            elif correlation_method == 'cross':
                corr_lag_results_full = timelag_cross_correlation_matrix(gait_cycle_data[all_markers], all_markers)
                correlation_matrix_full = corr_lag_results_full[0]
                timelag_matrix_full = corr_lag_results_full[1]
            else: # default - Pearson's
                correlation_matrix_full = np.array(gait_cycle_data[all_markers].corr(method='pearson', min_periods=1))
        
        else: # kinectomes for AP, ML and V directions separately
            # Initialize correlation matrices
            num_markers = len(marker_list)
            correlation_matrices = np.zeros((num_markers, num_markers, 3))
            timelag_matrices = np.zeros((num_markers, num_markers, 3))
            
            # Compute correlation for each coordinate (x, y, z)
            for i, coord in enumerate([f'_{K}_x', f'_{K}_y', f'_{K}_z']):
                markers = [m + coord for m in marker_list]
                if correlation_method == 'dcor':
                    correlation_matrices[:, :, i] = distance_correlation_matrix(gait_cycle_data[markers], markers)
                elif correlation_method == 'cross':
                    corr_lag_results = timelag_cross_correlation_matrix(gait_cycle_data[markers], markers)
                    correlation_matrices[:, :, i] = corr_lag_results[0]
                    timelag_matrices[:, :, i] = corr_lag_results[1]
                else: # default - Pearson's
                    correlation_matrices[:, :, i] = gait_cycle_data[markers].corr(method='pearson', min_periods=1)

        # directory to save
        kinectome_path = Path(base_path) / "derived_data" / f"sub-{sub_id}" / "kinectomes"
        kinectome_path.mkdir(parents=True, exist_ok=True)


        if full_kinectomes:
            # Define file name (_pos_ for kinetomes of marker position data, vel - velocity, acc - acceleration)
            if run: # 'run-off' or 'run-on' will appear in the kinectome file name
                if correlation_method == 'dcor':
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_dcor_full.npy"
                elif correlation_method == 'cross':
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_cross_full.npy"
                    file_name_timeLag = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_time_lag_full.npy"         
                else:
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_pears_full.npy"
            else: 
                if correlation_method == 'dcor':
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_dcor_full.npy"
                elif correlation_method == 'cross':
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_cross_full.npy"
                    file_name_timeLag = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_time_lag_full.npy"
                else:
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_pears_full.npy"
        else:
            # Define file name (_pos_ for kinetomes of marker position data, vel - velocity, acc - acceleration)
            if run: # 'run-off' or 'run-on' will appear in the kinectome file name
                if correlation_method == 'dcor':
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_dcor.npy"
                elif correlation_method == 'cross':
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_cross.npy"
                    file_name_timeLag = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_time_lag.npy"
                else:
                    file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_pears.npy"
            else: 
                if correlation_method == 'dcor':
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_dcor.npy"
                elif correlation_method == 'cross':
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_cross.npy"
                    file_name_timeLag = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_time_lag.npy"
                else:
                    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}_pears.npy"
        
        file_path = os.path.join(kinectome_path, file_name)
        
        if correlation_method == 'cross':
            file_path_timeLag = os.path.join(kinectome_path, file_name_timeLag)

        # insert the function which reorders the kinectome based on more and less affected sides here 

        demographics_row = find_demographics_row(sub_id, run)

        to_be_reordered = correlation_matrix_full if full_kinectomes else correlation_matrices

        reordered_correlation_matrix = reorder_kinectome_by_affected_side(to_be_reordered, marker_list, demographics_row, full_kinectomes)
        np.save(file_path, reordered_correlation_matrix[0])

        if correlation_method == 'cross':
            to_be_reordered_lag = timelag_matrix_full if full_kinectomes else timelag_matrices
            reordered_timelag_matrices = reorder_kinectome_by_affected_side(to_be_reordered_lag, marker_list, demographics_row, full_kinectomes)
            np.save(file_path_timeLag, reordered_timelag_matrices[0])

        # Save a copy to the local kinectome folder (if configured)
        from config import KINECTOME_SAVE_PATH
        if KINECTOME_SAVE_PATH is not None:
            local_path = Path(KINECTOME_SAVE_PATH) / f"sub-{sub_id}"
            local_path.mkdir(parents=True, exist_ok=True)
            np.save(local_path / file_name, reordered_correlation_matrix[0])
            if correlation_method == 'cross':
                np.save(local_path / file_name_timeLag, reordered_timelag_matrices[0])



def calculate_all_kinectomes(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, raw_data_path, fs, 
                             base_path, marker_list, result_base_path, full, correlation_method) -> None:
    """
    Calculates kinectomes for all subejcts. 
    This function iterates over a predefined list of subjects, tasks, tracking systems, and kinematic data types     
    to locate, load, preprocess, and analyze motion data files. Preprocessed data is then used to compute kinectomes. 

    Workflow:
        1. Make the disease (based on diagnosis variable) and matched control groups.
        2. Iterate through subjects, tasks, kinematics, and tracking systems to locate relevant motion files.
        3. Load motion tracking data from files.
        4. Preprocess data (trimming, dimension reduction, interpolation, rotation, differentiation).
        5. Compute kinectomes for each gait cycle and save them as `.npy` files. 

    Special Handling:
        - Subjects measured in the "on" medication condition may have filenames without explicit "run-on".
        - Control subjects (matched controls) do not have medication conditions and are processed with `run=None`. 

    Global Variables Used:
        - `diagnosid` (list): Specifies the disease of interest. 
        - `kinematics_list` (list): Types of kinematic data (e.g., position, velocity, acceleration).
        - `task_names` (list): Motion task names.
        - `tracking_systems` (list): Motion tracking systems used.
        - `runs` (list): Run conditions (e.g., "on", "off") for pwPD.
        - `raw_data_path` (str): Path to the raw motion data files.
        - `base_path` (str): Path to save computed kinectomes.
        - `fs` (float): Sampling frequency of motion data.
        - `marker_list` (list): List of markers used in motion tracking.

    Returns:
        None
    """
    from src.data_utils.bids_data_loader import BIDSDataLoader
    loader = BIDSDataLoader(base_path=base_path, raw_data_path=raw_data_path)

    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

    # use for debugging particular subjects
    debug_ids = ['pp053', 'pp091', 'pp117', 'pp128', 'pp140']

    # file name is based on task names and tracking systems defined in the global variables
 

    for sub_id in disease_sub_ids + matched_control_sub_ids:
    # for sub_id in debug_ids:
        for kinematics in kinematics_list:
            for task_name in task_names:
                for tracksys in tracking_systems:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'

                        # Skip only if kinectomes of the SAME full-ness already exist.
                        # Full kinectomes carry a "_full" suffix; direction-wise ones do not.
                        from config import KINECTOME_SAVE_PATH
                        local_dir = Path(KINECTOME_SAVE_PATH) / f"sub-{sub_id}"
                        run_token = run if run else ""

                        def _matches(f):
                            if not f.endswith(".npy"):
                                return False
                            if not (task_name in f and tracksys in f and kinematics in f
                                    and correlation_method in f):
                                return False
                            # full-ness must match: "_full" present iff we want full
                            is_full_file = "_full" in f
                            if is_full_file != bool(full):
                                return False
                            if run_token:
                                return run_token in f
                            return not any(f"run-{r}" in f for r in ["on", "off"])

                        already_done = local_dir.exists() and any(
                            _matches(p.name) for p in local_dir.iterdir()
                        )
                        if already_done:
                            print(f"  Skipping sub-{sub_id}, task-{task_name} — {'full ' if full else ''}kinectomes already exist.")
                            continue

                        # Load raw data via the configured data loader
                        data = loader.load_raw_data(sub_id, task_name, tracksys, run)

                        if data is None:
                            continue

                        # Detect gait events on the raw data (cone columns still present).
                        # This runs once and the result is passed into calculate_kinectome,
                        # where it is used by find_full_leftRight_cycles. Detection cannot
                        # happen later because preprocessing drops the cone marker columns.
                        try:
                            events = data_loader.detect_gait_events_from_data(data, fs=fs)
                        except (KeyError, ValueError) as e:
                            print(f"  Skipping sub-{sub_id}, task-{task_name}: {e}")
                            continue

                        if events is None or events.empty:
                            print(f"  Event detection failed for sub-{sub_id}, task-{task_name}. Skipping.")
                            continue

                        # Extract gate bounds from already-detected events
                        # so trim_to_walking_window doesn't re-run detection
                        _start = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
                        _stop  = int(events.loc[events['event_type'] == 'stop',  'onset'].values[0])

                        # Preprocess: trim (using pre-computed bounds), gap-fill, rotate, differentiate
                        preprocessed_data = all_preprocessing(
                            data, sub_id, task_name, run, tracksys, kinematics, fs,
                            frame_start=_start, frame_end=_stop
                        )

                        if preprocessed_data is None:
                            continue

                        # Controls have no run label in their filenames
                        if sub_id in matched_control_sub_ids:
                            run = None

                        # Compute and save kinectomes for every detected gait cycle
                        calculate_kinectome(
                            preprocessed_data, events, sub_id, task_name, run, tracksys,
                            kinematics, base_path, result_base_path,
                            marker_list, full, correlation_method
                        )

    return

def reorder_kinectome_by_affected_side(kinectome, marker_list, demographics_row, full):
    """Reorder kinectome markers by affected/less-affected side.

    Two modes, selected automatically based on available data:

    **UPDRS mode** (default for PD data with UPDRS III scores):
        The more-affected side is determined by summing UPDRS III scores for
        upper (3.3, 3.4, 3.5, 3.6) and lower (3.3, 3.7, 3.8) extremities.
        Requires columns ``updrs_3_3_rigidity_rue`` etc. in the demographics file.

    **Left/right mode** (fallback for datasets without UPDRS data):
        Right side is treated as less-affected (``_la``), left as more-affected
        (``_ma``).  This preserves the marker ordering logic without requiring
        clinical scores.  To swap the convention, set
        ``AFFECTED_SIDE_DEFAULT = 'right'`` in ``config.py``.

    Parameters
    ----------
    kinectome : np.ndarray
        Kinectome matrix of shape ``(n_markers, n_markers, 3)`` or
        ``(n_markers*3, n_markers*3)`` for full kinectomes.
    marker_list : list[str]
        Ordered marker names matching the kinectome rows/columns.
    demographics_row : pd.DataFrame
        One-row DataFrame for this subject from the demographics file.
    full : bool
        Whether this is a full (combined-direction) kinectome.

    Returns
    -------
    reordered_kinectome : np.ndarray
    reordered_labels : list[str]
    """
    from config import AFFECTED_SIDE_DEFAULT

    # Detect whether UPDRS columns are available
    updrs_cols = ['updrs_3_3_rigidity_rue', 'updrs_3_4_finger_taps_r',
                  'updrs_3_5_hand_movement_r', 'updrs_3_6_pro_sub_hand_r',
                  'updrs_3_3_rigidity_lue', 'updrs_3_4_finger_taps_l',
                  'updrs_3_5_hand_movement_l', 'updrs_3_6_pro_sub_hand_l',
                  'updrs_3_3_rigidity_rle', 'updrs_3_7_foot_tap_r', 'updrs_3_8_leg_agility_r',
                  'updrs_3_3_rigidity_lle', 'updrs_3_7_foot_tap_l', 'updrs_3_8_leg_agility_l']

    has_updrs = (demographics_row is not None
                 and not demographics_row.empty
                 and all(c in demographics_row.columns for c in updrs_cols)
                 and not demographics_row[updrs_cols].isnull().all().all())

    if has_updrs:
        # ── UPDRS mode ────────────────────────────────────────────────────────
        r_upper = demographics_row[['updrs_3_3_rigidity_rue', 'updrs_3_4_finger_taps_r',
                                     'updrs_3_5_hand_movement_r', 'updrs_3_6_pro_sub_hand_r']].sum().sum()
        l_upper = demographics_row[['updrs_3_3_rigidity_lue', 'updrs_3_4_finger_taps_l',
                                     'updrs_3_5_hand_movement_l', 'updrs_3_6_pro_sub_hand_l']].sum().sum()
        r_lower = demographics_row[['updrs_3_3_rigidity_rle', 'updrs_3_7_foot_tap_r',
                                     'updrs_3_8_leg_agility_r']].sum().sum()
        l_lower = demographics_row[['updrs_3_3_rigidity_lle', 'updrs_3_7_foot_tap_l',
                                     'updrs_3_8_leg_agility_l']].sum().sum()

        handedness_number = demographics_row.get('handedness').sum()
        if handedness_number == 0 or handedness_number == 999:
            handedness = 'right'
        elif handedness_number == 1:
            handedness = 'left'
        elif pd.isna(handedness_number):
            handedness = 'right'
        else:
            handedness = 'right'

    else:
        # ── Left/right mode (no UPDRS data available) ─────────────────────────
        print(f"  No UPDRS data found — using left/right convention "
              f"(affected={AFFECTED_SIDE_DEFAULT}). "
              f"Set AFFECTED_SIDE_DEFAULT in config.py to change.")
        # Treat AFFECTED_SIDE_DEFAULT as more-affected for both upper and lower
        upper_ma = AFFECTED_SIDE_DEFAULT
        upper_la = 'right' if AFFECTED_SIDE_DEFAULT == 'left' else 'left'
        lower_ma = upper_ma
        lower_la = upper_la


    def determine_more_affected(left_metric, right_metric, handedness):
        """
        Determine which side is more affected based on UPDRS III and handedness. 
        To determine the affectedness of the upper extremity, sum up the scores for each UE of:
        3.3 (rigidity), 3.4 (finger tapping), 3.5 (hand movements, 3.6 (hand pronation/supination))
        
        To determine the affectedness of the lower extremity, sum up the scores for each LE of:
        3.3 (rigidity), 3.7 (toe tapping), and 3.8 (leg agility).

        If the scores are equal, take the dominant hand as the less affected side;
        If there is no data on handedness, assume the person is right handed 
        (10.6% world's population is left handed DOI: 10.1037/bul0000229)

        Returns tuples of (more_affected, less_affected) sides.
        
        Args:
            left_metric: Metric value for left side
            right_metric: Metric value for right side
            handedness: Patient handedness ('left', 'right')
            
        Returns:
            Tuple of (more_affected, less_affected) as 'left' or 'right'
        """
        # Higher metric indicates more affected
        if left_metric > right_metric:
            return 'left', 'right'
        elif right_metric > left_metric:
            return 'right', 'left'
        else:
            # If metrics are equal, use handedness as a tiebreaker
            # Non-dominant side is considered more affected in a tie
            if handedness == 'right':
                return 'left', 'right'
            elif handedness == 'left':
                return 'right', 'left'
            else:  # ambidextrous
                # Default to left=more affected if ambidextrous and tied
                return 'left', 'right'

    if has_updrs:
        upper_ma, upper_la = determine_more_affected(l_upper, r_upper, handedness)
        lower_ma, lower_la = determine_more_affected(l_lower, r_lower, handedness)


    # Define which markers need to be relabeled
    def relabel(label):
        # Define which markers belong to upper/lower limbs
        upper_markers = ['sho', 'elbl', 'wrist', 'hand']
        lower_markers = ['asis', 'psis', 'th', 'sk', 'ank', 'toe']
        for marker in upper_markers:
            if f'l_{marker}' == label:
                return f'{marker}_ma' if upper_ma == 'left' else f'{marker}_la'
            if f'r_{marker}' == label:
                return f'{marker}_ma' if upper_ma == 'right' else f'{marker}_la'
        for marker in lower_markers:
            if f'l_{marker}' == label:
                return f'{marker}_ma' if lower_ma == 'left' else f'{marker}_la'
            if f'r_{marker}' == label:
                return f'{marker}_ma' if lower_ma == 'right' else f'{marker}_la'
        # If not a relabel target, return original
        return label
     
    # Desired order of markers without directional components
    desired_order = [
        'head', 'ster',
        'sho_la', 'sho_ma', 'elbl_la', 'elbl_ma', 'wrist_la', 'wrist_ma', 'hand_la', 'hand_ma',
        'asis_la', 'asis_ma', 'psis_la', 'psis_ma',
        'th_la', 'th_ma', 'sk_la', 'sk_ma', 'ank_la', 'ank_ma', 'toe_la', 'toe_ma'
    ]
    
    if not full:
        # Original 22x22x3 processing
        relabeled = [relabel(lbl) for lbl in marker_list]
        label_to_index = {label: i for i, label in enumerate(relabeled)}
        
        # Filter for only labels that exist in our data
        valid_order = [label for label in desired_order if label in label_to_index]
        sorted_idx = [label_to_index[label] for label in valid_order]
        
        reordered_kinectome = kinectome[sorted_idx][:, sorted_idx, :]
        reordered_labels = [relabeled[i] for i in sorted_idx]

        return reordered_kinectome, reordered_labels
    
    else:
        # Full 66x66 processing
        directions = ['AP', 'ML', 'V']
        
        # Create mapping from original markers to their position in MARKER_LIST
        marker_to_pos = {marker: i for i, marker in enumerate(marker_list)}
        
        # Relabel the markers
        relabeled_markers = [relabel(marker) for marker in marker_list]
        
        # Create mapping from relabeled markers to desired position
        desired_pos = {marker: i for i, marker in enumerate(desired_order)}
        
        # Create reordering index list (for all 66 indices)
        reorder_indices = []
        
        # For each marker and each direction in the desired order
        for marker in desired_order:
            # Find the original marker that matches this desired marker
            original_markers = []
            for i, relabeled in enumerate(relabeled_markers):
                if relabeled == marker:
                    original_markers.append(marker_list[i])
            
            if not original_markers:
                continue  # Skip if no match found
                
            for orig_marker in original_markers:
                orig_pos = marker_to_pos[orig_marker]
                # Add indices for all three directions (AP, ML, V)
                for direction_idx in range(3):
                    reorder_indices.append(orig_pos * 3 + direction_idx)
        
        # Reorder the kinectome using the computed indices
        reordered_kinectome = kinectome[reorder_indices, :][:, reorder_indices]
        
        # Generate new labels
        reordered_labels = []
        for marker in desired_order:
            for direction in directions:
                if any(relabeled == marker for relabeled in relabeled_markers):
                    reordered_labels.append(f"{marker}_{direction}")
        
        return reordered_kinectome, reordered_labels


def find_demographics_row(sub_id, run):
    from config import DEMOGRAPHICS_PATH
    demographics_path = Path(DEMOGRAPHICS_PATH)
    if demographics_path.suffix.lower() == '.csv':
        demographics_df = pd.read_csv(demographics_path)
    else:
        demographics_df = pd.read_excel(demographics_path)
    # e.g., if sub_id = 'pp008'
    numeric_id = int(sub_id[2:])  # Extract '008' → convert to 8
    # Match it in the demographics DataFrame
    subject_rows = demographics_df[demographics_df['id'] == numeric_id]

    if run is None:
        # Return row for control subject (no run condition)
        return subject_rows

    # For PD subjects, filter further by medication state
    # Assumes the column is named 'med_state' and values like 'ON' or 'OFF'
    run_str = str(run).upper()  # ensure consistent capitalization
    row = subject_rows[subject_rows['med_state'].str.upper() == run_str]

    return row


# NOTE: copy_kinectomes_to_local is a standalone utility not called from calculate_all_kinectomes
#       or any function in the active call chain. Commented out to keep the active code surface
#       clear. Uncomment to restore if needed.
#
import shutil
import os

def copy_kinectomes_to_local(result_base_path, local_destination_path):
    """
    Copies all computed kinectome .npy files from the server result path
    to a local destination (e.g., your personal PC or USB drive).
    
    Call this after calculate_all_kinectomes() has finished.
    
    Args:
        result_base_path (str): The server path where kinectomes were saved.
        local_destination_path (str): The local path to copy files to.
    """
    copied = 0
    skipped = 0

    for root, dirs, files in os.walk(result_base_path):
        for file in files:
            if file.endswith('.npy'):
                src = os.path.join(root, file)

                # Recreate the folder structure at the destination
                rel_path = os.path.relpath(src, result_base_path)
                dst = os.path.join(local_destination_path, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)

                # Skip if already copied and identical
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                    skipped += 1
                    continue

                shutil.copy2(src, dst)
                copied += 1

    print(f"Done. Copied: {copied} files | Skipped (already exist): {skipped} files")