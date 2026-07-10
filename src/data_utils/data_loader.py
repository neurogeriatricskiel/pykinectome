import pandas as pd
from pathlib import Path
import os
import re
import numpy as np

from src.data_utils.detect_gait_events import detect_events_from_dataframe


def load_file(file_path: str | Path) -> pd.DataFrame:
    """Load a tab-separated file into a DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the .tsv or .csv file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df


# def load_events(base_path, sub_id, task_name, run, linux=False):
#     """
#     [REPLACED by detect_gait_events_from_data]
#     Previously loaded pre-computed gait event .tsv files from disk.
#     Kept here for reference in case you need to revert or compare.
#
#     Load event data for a given subject, task, and run from a BIDS events .tsv.
#     Navigates to the subject's motion folder and selects the appropriate file
#     based on task name and run condition ('on'/'off').
#
#     Parameters
#     ----------
#     base_path : str or Path
#         Base directory containing rawdata/.
#     sub_id : str
#         Subject identifier (e.g. 'pp065').
#     task_name : str
#         Task name to match in the filename.
#     run : str
#         Run condition ('on' or 'off').
#     linux : bool, optional
#         If True, uses forward-slash path separators.
#
#     Returns
#     -------
#     pd.DataFrame
#         Events DataFrame with columns: onset, duration, event_type.
#     """
#     if linux:
#         os.chdir(f'{base_path}/rawdata/sub-{sub_id}/motion')
#     else:
#         os.chdir(f'{base_path}\\rawdata\\sub-{sub_id}\\motion')
#     file_list = os.listdir()
#     event_files = [file for file in file_list if task_name in file and 'events' in file]
#
#     if any(f"run-{r}" in file for r in ['on', 'off'] for file in event_files):
#         event_files = [file for file in event_files if f"run-{run}" in file and '.tsv' in file]
#     else:
#         event_files = [file for file in event_files if not any(
#             f"run-{r}" in file for r in ['on', 'off']) and '.tsv' in file]
#
#     events = load_file(event_files[0])
#     return events


def detect_gait_events_from_data(data: pd.DataFrame, fs: float,
                                  fc: float = 7.0,
                                  mm_to_m: bool = True) -> pd.DataFrame:
    """Detect gait events from raw motion capture data using Bonci et al. (2022).

    Replaces ``load_events()``.  Instead of reading pre-computed event files
    from disk, this function detects initial contacts (ICs) directly from the
    heel marker trajectories in the raw data DataFrame, using the walkway cone
    markers to gate the detection window.

    The returned DataFrame has exactly the same structure as the old events
    .tsv files, so the rest of the pipeline (``find_full_leftRight_cycles``)
    works without any changes.

    Parameters
    ----------
    data : pd.DataFrame
        Raw motion capture data for one subject/task/run, as loaded from the
        BIDS motion .tsv file.  Must contain columns for:

        - ``l_heel_POS_x``, ``l_heel_POS_y``, ``l_heel_POS_z``
        - ``r_heel_POS_x``, ``r_heel_POS_y``, ``r_heel_POS_z``
        - ``l_psis_POS_x``, ``r_psis_POS_x``  (for pelvis reference)
        - ``start_1_POS_x``, ``start_1_POS_y``, ``start_2_POS_x``, ``start_2_POS_y``
        - ``end_1_POS_x``,   ``end_1_POS_y``,   ``end_2_POS_x``,   ``end_2_POS_y``

    fs : float
        Sampling frequency in Hz (e.g. 200).
    fc : float, optional
        Low-pass filter cutoff frequency in Hz.  Default is 7 Hz, matching
        the original Bonci et al. (2022) implementation.
    mm_to_m : bool, optional
        Whether to convert marker positions from mm to m before processing.
        Default is True (OMC data is typically in mm).

    Returns
    -------
    pd.DataFrame
        Events table with columns ``onset``, ``duration``, ``event_type``,
        identical in structure to the old BIDS events .tsv files.
        ``onset`` values are 0-based frame indices into ``data``.
        ``event_type`` values: ``'start'``, ``'stop'``,
        ``'initial_contact_left'``, ``'initial_contact_right'``.

    Raises
    ------
    RuntimeError
        If IC detection fails (e.g. required marker columns are missing).

    Notes
    -----
    The detection runs on a temporary file written from ``data`` because
    ``load_and_process`` in the Bonci algorithm reads from disk.  The temp
    file is deleted after use.

    See Also
    --------
    src.data_utils.detect_gait_events : full Bonci et al. (2022) implementation.
    """
    results, frame_start, frame_end = detect_events_from_dataframe(
        data, fs=fs, fc=fc, mm_to_m=mm_to_m
    )

    # Build the events DataFrame in the same format as the old .tsv files
    rows = [
        {'onset': frame_start, 'duration': 0, 'event_type': 'start'},
        {'onset': frame_end,   'duration': 0, 'event_type': 'stop'},
    ]
    for side in ['left', 'right']:
        if side not in results:
            continue
        label = f'initial_contact_{side}'
        for idx in results[side]['IC']:
            rows.append({'onset': int(idx), 'duration': 0, 'event_type': label})

    events_df = (pd.DataFrame(rows)
                 .sort_values('onset')
                 .reset_index(drop=True))

    return events_df


def load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics,
                    full, correlation_method, interpol=False):
    """Load pre-computed kinectome .npy files for a subject and sort by onset.

    Parameters
    ----------
    base_path : str or Path
        Root data directory containing ``derived_data/``.
    sub_id : str
        Subject identifier (e.g. 'pp065').
    task_name : str
        Task name (e.g. 'walkStroop').
    tracksys : str
        Tracking system (e.g. 'omc').
    run : str or None
        Run condition ('on', 'off', or None for controls).
    kinematics : str
        Kinematic type (e.g. 'acc').
    full : bool
        Whether to load full (combined-direction) kinectomes.
    correlation_method : str
        Correlation method used to build the kinectomes (e.g. 'pears').
    interpol : bool, optional
        Whether to load kinectomes built from interpolated data.

    Returns
    -------
    list[np.ndarray] or None
        List of kinectome arrays sorted by gait cycle onset, or None if no
        matching files are found.
    """
    try:
        # base_path is KINECTOME_SAVE_PATH — files are flat in sub-<id>/
        kinectome_dir = Path(base_path) / f'sub-{sub_id}'

        if not kinectome_dir.exists():
            return None

        file_list = [f.name for f in kinectome_dir.iterdir() if f.suffix == '.npy']

        # Build base token list — tokens that must all appear in the filename
        base_tokens = [task_name, tracksys, kinematics, correlation_method]
        if full:
            base_tokens.append('full')
        # Note: interpol flag controls gait cycle resampling during computation
        # but is NOT encoded in the filename — gap-filling always runs in preprocessing

        # Exclusion tokens — must NOT appear
        excl = []
        if not full:
            excl.append('full')

        def matches(f, run_token=None):
            tokens = base_tokens + ([run_token] if run_token else [])
            return (all(x in f for x in tokens)
                    and all(x not in f for x in excl)
                    and (not run_token or not any(f"run-{r}" in f for r in ['on', 'off'] if r != run_token)))

        if run:
            # Try with run token first (PD subjects)
            relevant_files = [f for f in file_list if matches(f, run_token=run)]
            # Fall back to no run token (in case subject was saved without it)
            if not relevant_files:
                relevant_files = [f for f in file_list if matches(f)
                                  and not any(f"run-{r}" in f for r in ['on', 'off'])]
        else:
            # Controls — no run token in filename
            relevant_files = [f for f in file_list if matches(f)
                              and not any(f"run-{r}" in f for r in ['on', 'off'])]
            # Fall back: if nothing found without run token, try with any run token
            if not relevant_files:
                relevant_files = [f for f in file_list if matches(f)]

        sorted_files = sorted(relevant_files, key=lambda f: extract_onset_indices(f)[0])

        if not sorted_files:
            return None

        return [np.load(kinectome_dir / f) for f in sorted_files]

    except (FileNotFoundError, OSError):
        return None


def extract_onset_indices(filename):
    """Extract numerical gait cycle onset indices from a kinectome filename.

    Kinectome files are named with ``kinct<start>-<end>`` to encode the gait
    cycle boundaries.  This function parses those indices for sorting.

    Parameters
    ----------
    filename : str
        Kinectome filename (e.g.
        ``sub-pp001_task-walkStroop_kinct1234-1456_acc_pears_interpol.npy``).

    Returns
    -------
    tuple[int, int] or tuple[None, None]
        ``(start_frame, end_frame)`` indices, or ``(None, None)`` if the
        pattern is not found.
    """
    match = re.search(r'kinct(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def merge_dicts(list_of_dicts):
    """Merge a list of dictionaries into a single dict with list values.

    Parameters
    ----------
    list_of_dicts : list[dict]
        Each dict must have the same keys.

    Returns
    -------
    collections.defaultdict[list]
        Dictionary mapping each key to a list of values from all input dicts.
    """
    import collections
    result = collections.defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            result[key].append(value)
    return result


def exclude_markers_from_kinectome(kinectome: np.ndarray, marker_list: list,
                                    exclude: list) -> tuple:
    """Remove specified markers from a kinectome matrix.

    Drops the rows and columns corresponding to excluded markers from the
    kinectome array.  Works for both 2D (full kinectome) and 3D
    (n_markers × n_markers × 3 directions) arrays.

    Parameters
    ----------
    kinectome : np.ndarray
        Kinectome matrix of shape ``(n, n)`` or ``(n, n, 3)``.
    marker_list : list[str]
        Ordered marker names corresponding to kinectome rows/columns.
    exclude : list[str]
        Marker names to remove.

    Returns
    -------
    kinectome_reduced : np.ndarray
        Kinectome with excluded markers removed.
    marker_list_reduced : list[str]
        Updated marker list with excluded markers removed.
    """
    keep_idx = [i for i, m in enumerate(marker_list) if m not in exclude]
    marker_list_reduced = [marker_list[i] for i in keep_idx]

    if kinectome.ndim == 2:
        kinectome_reduced = kinectome[np.ix_(keep_idx, keep_idx)]
    else:
        kinectome_reduced = kinectome[np.ix_(keep_idx, keep_idx,
                                             list(range(kinectome.shape[2])))]
    return kinectome_reduced, marker_list_reduced