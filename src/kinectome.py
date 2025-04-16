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

def find_gait_cycles(base_path, data: pd.DataFrame, sub_id: str, task_name: str, run: str, linux: bool):
    """
    Identifies full left and right gait cycles based on event markers within the valid time range.

    Parameters:
    - data (pd.DataFrame): The motion tracking data.
    - sub_id (str): Subject identifier.
    - task_name (str): Name of the task performed.
    - run (str): Specifies the run condition (e.g., "on" or "off" for PD subjects).
    - linux (bool): Flag indicating whether the code is run on a Linux system.

    Returns:
    - gait_cycles (list of tuples): List of (start, end) indices for each detected gait cycle.
    - start_onset (int): The index corresponding to the start of the valid motion tracking period.
    """

    events = data_loader.load_events(base_path, sub_id, task_name, run)

    start_onset = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
    stop_onset = int(events.loc[events['event_type'] == 'stop', 'onset'].values[0])

    # Adjust events indices to match trimmed data (which starts at 0)
    events['onset'] = events['onset'] - start_onset

    # Ensure only events within start and stop are considered
    valid_events = events[(events['onset'] >= 0) & (events['onset'] <= (stop_onset - start_onset))]

    # Find initial contact left (ICL) and initial contact right (ICR) events
    icl_indices = valid_events[valid_events['event_type'] == 'initial_contact_left']['onset'].values
    icr_indices = valid_events[valid_events['event_type'] == 'initial_contact_right']['onset'].values

    # Iterate over gait cycles (left and right)
    gait_cycles = []

    for i in range(len(icl_indices) - 1):
        start_cycle = icl_indices[i]
        end_cycle = icl_indices[i + 1]  # Next ICL marks end of current cycle

        if start_cycle >= 0 and end_cycle <= len(data):
            gait_cycles.append((int(start_cycle), int(end_cycle)))


    for i in range(len(icr_indices) - 1):
        start_cycle = icr_indices[i]
        end_cycle = icr_indices[i + 1]  # Next ICR marks end of current cycle

        if start_cycle >= 0 and end_cycle <= len(data):
            gait_cycles.append((int(start_cycle), int(end_cycle)))       

    # Sort gait cycles by start index
    gait_cycles = sorted(gait_cycles, key=lambda x: x[0])

    return gait_cycles, start_onset


def find_full_leftRight_cycles(base_path, data: pd.DataFrame, sub_id: str, task_name: str, run: str):
    """
    Identifies gait cycles that include both a full right and a full left gait cycle.

    Parameters:
    - base_path (str): Base directory path.
    - data (pd.DataFrame): The motion tracking data.
    - sub_id (str): Subject identifier.
    - task_name (str): Name of the task performed.
    - run (str): Specifies the run condition (e.g., "on" or "off" for PD subjects).

    Returns:
    - gait_cycles (list of tuples): List of (start, end) indices for each full gait cycle.
    - start_onset (int): The index corresponding to the start of the valid motion tracking period.
    """

    events = data_loader.load_events(base_path, sub_id, task_name, run)

    start_onset = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
    stop_onset = int(events.loc[events['event_type'] == 'stop', 'onset'].values[0])

    # Adjust event indices relative to the trimmed data (starting at 0)
    events['onset'] = events['onset'] - start_onset

    # Ensure only events within the valid start-stop range are considered
    valid_events = events[(events['onset'] >= 0) & (events['onset'] <= (stop_onset - start_onset))]

    icl_indices = valid_events[valid_events['event_type'] == 'initial_contact_left']['onset'].values
    icr_indices = valid_events[valid_events['event_type'] == 'initial_contact_right']['onset'].values

    gait_cycles = []

    i, j = 0, 0

    while i < len(icr_indices) - 1 and j < len(icl_indices) - 1:
        start_cycle = icr_indices[i]  # Start at right initial contact
        end_cycle = None

        # Find the second left initial contact after the right initial contact
        for j in range(len(icl_indices)):
            if icl_indices[j] > start_cycle:
                if j + 1 < len(icl_indices):  # Ensure we can get the next left contact
                    end_cycle = icl_indices[j + 1]
                    break

        if end_cycle and start_cycle >= 0 and end_cycle <= len(data):
            gait_cycles.append((int(start_cycle), int(end_cycle)))

        i += 1  # Move to the next right initial contact

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

def calculate_kinectome(data: pd.DataFrame, sub_id: str, task_name: str, run: str, tracksys: str, kinematics: str, base_path: str, result_base_path: str, marker_list: list, 
                        full_kinectomes, 
                        correlation_method,
                        linux = False, 
                        ):
    """
    Computes Pearson correlation matrices for marker positions in x, y, and z coordinates across gait cycles 
    and saves them as .npy files.

    Parameters:
    - data (pd.DataFrame): The motion tracking data.
    - sub_id (str): Subject identifier.
    - task_name (str): Name of the task performed.
    - run (str): Specifies the run condition (e.g., "on" or "off").
    - tracksys (str): The tracking system used for data collection.
    - base_path (str): Base directory where the kinectome data should be saved.
    - kinematics (str): Marker position or its derivatives (velocity, acceleration) used for calculating kinectomes.
    - linux (bool): Flag indicating whether the code is run on a Linux system.
    - dcor (bool): Flag indicating whether to use distance correlation.
    - crosscorr (bool): Flag indicating whether to use time-lag corss correlation.

    Returns:
    - None: The function saves correlation matrices but does not return a value.
    """

    gait_cycles, start_onset = find_full_leftRight_cycles(base_path, data, sub_id, task_name, run)
    cycles_iterator = tqdm(gait_cycles, desc=f"---Subject: {sub_id}, Task: {task_name}---")
    for i, cycles in enumerate(cycles_iterator): #range(len(gait_cycles)):
        cycle_indices = gait_cycles[i]

        gait_cycle_data = segment_data(data, cycle_indices)

        # Extract marker names
        marker_names = sorted(set(col[:-6] for col in gait_cycle_data.columns if col.endswith(f'{kinematics.upper()}_x')))

        # Check for missing markers
        missing_markers = [m for m in marker_list if m not in marker_names]
        if missing_markers:
            print(f"Missing markers for Subject: {sub_id}, Task: {task_name}, Missing: {missing_markers}")
            continue

        # Reorder columns based on MARKER_LIST
        ordered_columns = []
        for marker in marker_list:
            if marker in marker_names:
                ordered_columns.extend([f"{marker}_{kinematics.upper()}_x", 
                                        f"{marker}_{kinematics.upper()}_y", 
                                        f"{marker}_{kinematics.upper()}_z"])

        # Subset and reorder dataframe
        gait_cycle_data = gait_cycle_data[ordered_columns]

        
        # compute correlations for all coordinates (x AND y AND z)        
        if full_kinectomes: 
            num_markers = len(marker_names)       
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
            num_markers = len(marker_names)
            correlation_matrices = np.zeros((num_markers, num_markers, 3))
            timelag_matrices = np.zeros((num_markers, num_markers, 3))
            
            # Compute correlation for each coordinate (x, y, z)
            for i, coord in enumerate([f'_{kinematics.upper()}_x', f'_{kinematics.upper()}_y', f'_{kinematics.upper()}_z']):
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
        if linux:
            kinectome_path = f"{base_path}/derived_data/sub-{sub_id}/kinectomes"
        else:
            kinectome_path = f"{base_path}\\derived_data\\sub-{sub_id}\\kinectomes"

        # Ensure directory exists
        if not os.path.exists(kinectome_path):
            os.makedirs(kinectome_path)


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
    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

    # use for debugging particular subjects
    debug_ids = ['pp032']

    # file name is based on task names and tracking systems defined in the global variables
 

    for sub_id in disease_sub_ids + matched_control_sub_ids:
    # for sub_id in debug_ids:
        for kinematics in kinematics_list: 
            for task_name in task_names:
                for tracksys in tracking_systems:
                    for run in runs:
                        if sub_id in pd_on: # those sub ids which are measured in 'on' condition but there is no 'run-on' in the filename
                            run = 'on'
                        else:
                            run = run                      
                        # change current working directory to the folder with motion data
                        os.chdir(f'{raw_data_path}\\sub-{sub_id}\\motion')
                        
                        file_list = os.listdir()
 
                        file_path = None  # Initialize file_path for each loop
 
                        for file in file_list:
                            if sub_id in file and task_name in file and tracksys in file and 'motion' in file:
                                # Check if the 'run' condition matches 
                                if f'run-{run}' in file:
                                    file_path = f"{raw_data_path}\\sub-{sub_id}\\motion\\{file}"
                                    break # Exit the loop once a matching file is found
 
                                # Include files without any 'run-on' or 'run-off' condition (run-on and run-off are only applicable to PD)

                                elif not any(f"run-{cond}" in file for cond in ["on", "off"]):
                                    file_path = f"{raw_data_path}\\sub-{sub_id}\\motion\\{file}"
                                    break
 
                        if file_path:
                            # Load the data as a pandas dataframe
                            data = data_loader.load_file(file_path=file_path) 
                            
                            # trim, reduce dimensions, interpolate, rotate, differentiate
                            preprocessed_data = all_preprocessing(data, sub_id, task_name, run, tracksys, kinematics, fs)
 
                            if preprocessed_data is None:
                                continue                                
 
                            # Calculate kinectomes (for each gait cycle) and save in derived_data/kinectomes
                            # can be done only once for each derivative (position, velocity, acceleration etc.) and then commented out to save on running time

                            if sub_id in matched_control_sub_ids: # pwPD that were measured on medication and have no 'run' in the filename
                                run = None         

                            # calculates the kinectomes in AP, ML and V directions and saves as .npy files
                            calculate_kinectome(preprocessed_data, sub_id, task_name, run, tracksys, kinematics, base_path, result_base_path, 
                                                marker_list, full, correlation_method)
                        else:
                            print(f"No matching motion file found for sub-{sub_id}, task-{task_name}, tracksys-{tracksys}, run-{run}")
 

    return

def reorder_kinectome_by_affected_side(kinectome, marker_list, demographics_row, full):
    """
    Reorder a (33, 33, 3) kinectome based on more/less affected sides using UPDRS scores.

    Parameters:
    - kinectome: np.array of shape (33, 33, 3)
    - labels: list of str, current marker labels in the correct order
    - demographics_row: pd.Series with UPDRS scores for one subject

    Returns:
    - reordered_kinectome: np.array of shape (33, 33, 3)
    - reordered_labels: list of str
    """

    # Sum scores for each limb group
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
    elif pd .isna(handedness_number):
        handedness = 'right'


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
    demographics = pd.read_excel(Path("Z:\\Keep Control\\Data\\demographics_scores_internal_use_only.xlsx"))
    demographics_df = pd.DataFrame(demographics)
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