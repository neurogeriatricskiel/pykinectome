import os
import pandas as pd
from src.data_utils import data_loader

def startStop(data: pd.DataFrame, sub_id: str, task_name: str, run: str) -> pd.DataFrame:
    """
    Extracts a subset of the given DataFrame based on start and stop event onsets.

    This function searches for an event file in the current directory that matches
    the specified task name. If the 'run' condition is relevant, it ensures the correct
    file is selected. It then loads the event file to determine the onset times for
    the "start" and "stop" events and trims the provided DataFrame accordingly.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be trimmed.
        sub_id (str): The subject identifier.
        task_name (str): The name of the task to identify the relevant event file.
        run (str): The run identifier to further specify the event file (only valid for PD subjects).

    Returns:
        pd.DataFrame: A subset of the input DataFrame containing data between the
                      detected start and stop onsets.

    Raises:
        FileNotFoundError: If no matching event file is found.
        ValueError: If the start or stop event is not found in the event file.
    """
    
    file_list = os.listdir()
    event_files = [file for file in file_list if task_name in file and 'events' in file]

     # Filter event files based on 'run' condition
    if any(f"run-{r}" in file for r in ['on', 'off'] for file in event_files):
        event_files = [file for file in event_files if f"run-{run}" in file and '.tsv' in file]
    else:
        event_files = [file for file in event_files if not any(f"run-{r}" in file for r in ['on', 'off']) and '.tsv' in file]
    
    if not event_files:
        print(f"Warning: No events file found for subject {sub_id}, task {task_name} during run-{run}. Skipping...")
        return None  # Return None instead of raising an error

    
    events = data_loader.load_file(event_files[0])

    try:
        start_onset = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
        stop_onset = int(events.loc[events['event_type'] == 'stop', 'onset'].values[0])

    except IndexError:
        print(f"Start or stop event missing for subject {sub_id}, task {task_name}")
        return None
    
    # Cut the data to be between start and stop
    trimmed_data = data[start_onset:stop_onset]

    # remove the start and marker positions
    trimmed_data = trimmed_data.drop(columns=data.filter(regex=r'(start_)\d+_POS_(x|y|z|err)').columns)
    trimmed_data = trimmed_data.drop(columns=data.filter(regex=r'(end_)\d+_POS_(x|y|z|err)').columns)

    return trimmed_data 

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
