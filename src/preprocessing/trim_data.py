import os
import pandas as pd
from src.data_utils import data_loader


# def startStop(data, sub_id, task_name, run):
#     """
#     [REPLACED by trim_to_walking_window]
#     Previously loaded pre-computed event .tsv files from the current working
#     directory (set by os.chdir in calculate_all_kinectomes) to find the start
#     and stop frame indices, then trimmed the data accordingly.
#     Kept here for reference.
#
#     Equivalent new call:
#         trim_to_walking_window(data, sub_id, task_name, run, mode="cone", fs=FS)
#     """
#     file_list = os.listdir()
#     event_files = [f for f in file_list if task_name in f and 'events' in f]
#     if any(f"run-{r}" in f for r in ['on', 'off'] for f in event_files):
#         event_files = [f for f in event_files if f"run-{run}" in f and '.tsv' in f]
#     else:
#         event_files = [f for f in event_files
#                        if not any(f"run-{r}" in f for r in ['on', 'off']) and '.tsv' in f]
#     if not event_files:
#         print(f"Warning: No events file found for {sub_id}, {task_name}, run-{run}.")
#         return None
#     events = data_loader.load_file(event_files[0])
#     try:
#         start = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
#         stop  = int(events.loc[events['event_type'] == 'stop',  'onset'].values[0])
#     except IndexError:
#         print(f"Start or stop event missing for {sub_id}, {task_name}")
#         return None
#     trimmed = data[start:stop]
#     trimmed = trimmed.drop(columns=data.filter(regex=r'(start_)\d+_POS_(x|y|z|err)').columns)
#     trimmed = trimmed.drop(columns=data.filter(regex=r'(end_)\d+_POS_(x|y|z|err)').columns)
#     return trimmed


def trim_to_walking_window(data: pd.DataFrame, sub_id: str, task_name: str,
                            run: str, mode: str = "cone",
                            fs: float = 200,
                            frame_start: int = None,
                            frame_end: int = None) -> pd.DataFrame:
    """Trim raw motion capture data to the active walking window.

    Supports two modes (set ``TRIM_MODE`` in ``config.py``):

    ``"cone"``
        Detects the start and stop of the walking window automatically from
        walkway cone markers in the raw data using the Bonci et al. (2022)
        algorithm (``find_walkway_bounds``).  Cone marker columns
        (``start_1/2``, ``end_1/2``) are dropped from the output since they
        are no longer needed after trimming.

    ``"none"``
        No trimming applied; the full DataFrame is returned unchanged.
        Use when data is already pre-cropped, or when the entire recording
        should be analysed.

    Parameters
    ----------
    data : pd.DataFrame
        Raw motion capture data.  For ``mode="cone"``, must contain columns:
        ``start_1_POS_x/y``, ``start_2_POS_x/y``, ``end_1_POS_x/y``,
        ``end_2_POS_x/y``, ``l_heel_POS_x/y``, ``r_heel_POS_x/y``.
    sub_id : str
        Subject identifier (used for logging).
    task_name : str
        Task name (used for logging).
    run : str
        Run condition (used for logging).
    mode : str, optional
        Trimming mode: ``"cone"`` or ``"none"``.  Default ``"cone"``.
    fs : float, optional
        Sampling frequency in Hz.  Required for ``mode="cone"``.

    Returns
    -------
    pd.DataFrame or None
        Trimmed DataFrame with cone columns removed (``mode="cone"``),
        the original DataFrame (``mode="none"``), or ``None`` on failure.
    """
    if mode == "none":
        return data

    if mode == "cone":
        # Reuse pre-computed bounds if provided (avoids running detection twice)
        if frame_start is None or frame_end is None:
            from src.data_utils.detect_gait_events import find_walkway_bounds
            try:
                frame_start, frame_end = find_walkway_bounds(data, mm_to_m=True, fs=fs)
            except Exception as e:
                print(f"Warning: cone detection failed for {sub_id}, {task_name}, "
                      f"run-{run}: {e}. Skipping.")
                return None

        trimmed = data.iloc[frame_start:frame_end].copy()

        # Drop cone marker columns — no longer needed after trimming
        trimmed = trimmed.drop(
            columns=data.filter(regex=r'(start_)\d+_POS_(x|y|z|err)').columns,
            errors='ignore'
        )
        trimmed = trimmed.drop(
            columns=data.filter(regex=r'(end_)\d+_POS_(x|y|z|err)').columns,
            errors='ignore'
        )
        trimmed = trimmed.reset_index(drop=True)
        return trimmed

    raise ValueError(
        f"Unknown TRIM_MODE '{mode}'. Expected 'cone' or 'none'. "
        "Add custom modes in src/preprocessing/trim_data.py."
    )



import pandas as pd

def reduce_dimensions_clusters(data: pd.DataFrame, sub_id: str, task_name: str) -> pd.DataFrame:
    """ 4-marker clusters of thighs and shanks are calculated into one midpoint,  
    3 sternum markers calculated into one,
    4 head markers into one midpoint,
    2 wrist markers into one midpoint
    
    """

    try:
        # Ensure a copy to avoid modifying a slice
        data = data.copy()

        # Calculate averages and create new columns using .loc
        data.loc[:, 'l_th_POS_x'] = data.filter(regex=r'l_th\d+_POS_x').mean(axis=1)
        data.loc[:, 'l_th_POS_y'] = data.filter(regex=r'l_th\d+_POS_y').mean(axis=1)
        data.loc[:, 'l_th_POS_z'] = data.filter(regex=r'l_th\d+_POS_z').mean(axis=1)

        data.loc[:, 'r_th_POS_x'] = data.filter(regex=r'r_th\d+_POS_x').mean(axis=1)
        data.loc[:, 'r_th_POS_y'] = data.filter(regex=r'r_th\d+_POS_y').mean(axis=1)
        data.loc[:, 'r_th_POS_z'] = data.filter(regex=r'r_th\d+_POS_z').mean(axis=1)

        data.loc[:, 'l_sk_POS_x'] = data.filter(regex=r'l_sk\d+_POS_x').mean(axis=1)
        data.loc[:, 'l_sk_POS_y'] = data.filter(regex=r'l_sk\d+_POS_y').mean(axis=1)
        data.loc[:, 'l_sk_POS_z'] = data.filter(regex=r'l_sk\d+_POS_z').mean(axis=1)

        data.loc[:, 'r_sk_POS_x'] = data.filter(regex=r'r_sk\d+_POS_x').mean(axis=1)
        data.loc[:, 'r_sk_POS_y'] = data.filter(regex=r'r_sk\d+_POS_y').mean(axis=1)
        data.loc[:, 'r_sk_POS_z'] = data.filter(regex=r'r_sk\d+_POS_z').mean(axis=1)

        data.loc[:, 'ster_POS_x'] = data.filter(regex=r'm_ster\d+_POS_x').mean(axis=1)
        data.loc[:, 'ster_POS_y'] = data.filter(regex=r'm_ster\d+_POS_y').mean(axis=1)
        data.loc[:, 'ster_POS_z'] = data.filter(regex=r'm_ster\d+_POS_z').mean(axis=1)

        data.loc[:, 'head_POS_x'] = data.filter(regex=r'^[lr][bf]_hd_POS_x$').mean(axis=1)
        data.loc[:, 'head_POS_y'] = data.filter(regex=r'^[lr][bf]_hd_POS_y$').mean(axis=1)
        data.loc[:, 'head_POS_z'] = data.filter(regex=r'^[lr][bf]_hd_POS_z$').mean(axis=1)

        data.loc[:, 'l_wrist_POS_x'] = data[['l_wrr_POS_x', 'l_wru_POS_x']].mean(axis=1)
        data.loc[:, 'l_wrist_POS_y'] = data[['l_wrr_POS_y', 'l_wru_POS_y']].mean(axis=1)
        data.loc[:, 'l_wrist_POS_z'] = data[['l_wrr_POS_z', 'l_wru_POS_z']].mean(axis=1)

        data.loc[:, 'r_wrist_POS_x'] = data[['r_wrr_POS_x', 'r_wru_POS_x']].mean(axis=1)
        data.loc[:, 'r_wrist_POS_y'] = data[['r_wrr_POS_y', 'r_wru_POS_y']].mean(axis=1)
        data.loc[:, 'r_wrist_POS_z'] = data[['r_wrr_POS_z', 'r_wru_POS_z']].mean(axis=1)


        # Drop original columns
        data = data.drop(columns=data.filter(regex=r'(l_th|r_th|l_sk|r_sk)\d+_POS_(x|y|z)').columns)
        data = data.drop(columns=data.filter(regex=r'[lr][bf]_hd_POS_[xyz]$').columns)   
        data = data.drop(columns=data.filter(regex=r'(m_ster)\d+_POS_(x|y|z)').columns)
        data = data.drop(columns=data.filter(regex='_err').columns)
        data = data.drop(columns=data.filter(regex=r'_(wrr|wru)(?!st)').columns)
        data = data.drop(columns=data.filter(regex=r'_ua_POS_[xyz]$'))
        data = data.drop(columns=data.filter(regex=r'_frm_POS_[xyz]$'))
        data = data.drop(columns=data.filter(regex=r'_heel_POS_[xyz]$'))


        return data

    except KeyError as e:
        print(f"Missing key(s) in DataFrame: {e} for subject {sub_id} during {task_name}")
        return None


def reduce_dimensions_hip(data: pd.DataFrame):
     # first fill the gaps, then calculate the midpoint? or first midpoint, then gaps? but then there are no gaps :/ lame 
        data['r_hip_POS_x'] = data[['r_asis_POS_x', 'r_psis_POS_x']].mean(axis=1)
        data['r_hip_POS_y'] = data[['r_asis_POS_y', 'r_psis_POS_y']].mean(axis=1)
        data['r_hip_POS_z'] = data[['r_asis_POS_z', 'r_psis_POS_z']].mean(axis=1)
        
        data['l_hip_POS_x'] = data[['l_asis_POS_x', 'l_psis_POS_x']].mean(axis=1)
        data['l_hip_POS_y'] = data[['l_asis_POS_y', 'l_psis_POS_y']].mean(axis=1)
        data['l_hip_POS_z'] = data[['l_asis_POS_z', 'l_psis_POS_z']].mean(axis=1)

        data = data.drop(columns=data.filter(regex='sis_POS_[xyz]').columns)

        return data

def remove_long_nans(data: pd.DataFrame, sub_id, task_name, run, nan_threshold=300):
    """
    Removes long NaN streaks (> nan_threshold) from the start or end of the dataframe.
    If long NaN streaks are in the middle, it ensures that full gait cycles are preserved.

    Args:
        data (pd.DataFrame): Motion tracking data with NaNs.
        nan_threshold (int): The minimum NaN streak length to be considered for removal.

    Returns:
        trimmed_data (pd.DataFrame): Data after trimming long NaN streaks.
        index_shift (int): The amount by which the indices were shifted.
    """
    
    max_nan_streak = 0
    streak_idx = None 

    for col in data.columns:
        # Convert NaN to 1, non-NaN to 0
        is_nan = data[col].isna().astype(int)
        
        # Create groups of consecutive NaN values
        streak_groups = (is_nan != is_nan.shift()).cumsum()
        
        # Calculate streak lengths for each group
        streak_lengths = is_nan.groupby(streak_groups).cumsum()
        
        # Find the maximum streak in this column
        max_streak = streak_lengths.max()

        if max_streak is None or max_nan_streak == 0:
            return data, None
        
        elif max_streak > max_nan_streak:            
            max_nan_streak = max_streak
            nan_col = col
            
            # Find the group with the longest streak
            max_group = streak_groups[streak_lengths == max_streak].iloc[0]
            # Get indices where this group starts and ends        
            streak_idx = (streak_lengths[streak_groups == max_group].index[0], streak_lengths[streak_groups == max_group].index[-1])
   
    if streak_idx[1] - streak_idx[0] > 400:
        print(f'{sub_id} has {streak_idx[1] - streak_idx[0]} NaNs in {nan_col} during {task_name}')
    
    # cut the data leaving the NaNs out

    ## how to cut the data so 
    return data, streak_idx