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
        print(f"Warning: No events file found for subject {sub_id}, task {task_name}. Skipping...")
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

    return trimmed_data 

def reduce_dimensions(data: pd.DataFrame, sub_id: str, task_name: str) -> pd.DataFrame:
    """ 4-marker clusters are calculated into one midpoint, 
    2 same side head markers calculated into one, 
    3 sternum markers calculated into one,
    2 same side hip markers calculated into one"""


    try:
        # Calculate averages and create new columns
        data['l_th_POS_x'] = data.filter(regex=r'l_th\d+_POS_x').mean(axis=1)
        data['l_th_POS_y'] = data.filter(regex=r'l_th\d+_POS_y').mean(axis=1)
        data['l_th_POS_z'] = data.filter(regex=r'l_th\d+_POS_z').mean(axis=1)

        data['r_th_POS_x'] = data.filter(regex=r'r_th\d+_POS_x').mean(axis=1)
        data['r_th_POS_y'] = data.filter(regex=r'r_th\d+_POS_y').mean(axis=1)
        data['r_th_POS_z'] = data.filter(regex=r'r_th\d+_POS_z').mean(axis=1)

        data['l_sk_POS_x'] = data.filter(regex=r'l_sk\d+_POS_x').mean(axis=1)
        data['l_sk_POS_y'] = data.filter(regex=r'l_sk\d+_POS_y').mean(axis=1)
        data['l_sk_POS_z'] = data.filter(regex=r'l_sk\d+_POS_z').mean(axis=1)

        data['r_sk_POS_x'] = data.filter(regex=r'r_sk\d+_POS_x').mean(axis=1)
        data['r_sk_POS_y'] = data.filter(regex=r'r_sk\d+_POS_y').mean(axis=1)
        data['r_sk_POS_z'] = data.filter(regex=r'r_sk\d+_POS_z').mean(axis=1)

        data['ster_POS_x'] = data.filter(regex=r'm_ster\d+_POS_x').mean(axis=1)
        data['ster_POS_y'] = data.filter(regex=r'm_ster\d+_POS_y').mean(axis=1)
        data['ster_POS_z'] = data.filter(regex=r'm_ster\d+_POS_z').mean(axis=1)

        data['l_head_POS_x'] = data[['lf_hd_POS_x', 'lb_hd_POS_x']].mean(axis=1)
        data['l_head_POS_y'] = data[['lf_hd_POS_y', 'lb_hd_POS_y']].mean(axis=1)
        data['l_head_POS_z'] = data[['lf_hd_POS_z', 'lb_hd_POS_z']].mean(axis=1)

        data['r_head_POS_x'] = data[['rf_hd_POS_x', 'rb_hd_POS_x']].mean(axis=1)
        data['r_head_POS_y'] = data[['rf_hd_POS_y', 'rb_hd_POS_y']].mean(axis=1)
        data['r_head_POS_z'] = data[['rf_hd_POS_z', 'rb_hd_POS_z']].mean(axis=1)


        # first fill the gaps, then calculate the midpoint? or first midpoint, then gaps? but then there are no gaps :/ lame 
        data['r_hip_POS_x'] = data[['r_asis_POS_x', 'r_psis_POS_x']].mean(axis=1)
        data['r_hip_POS_y'] = data[['r_asis_POS_y', 'r_psis_POS_y']].mean(axis=1)
        data['r_hip_POS_z'] = data[['r_asis_POS_z', 'r_psis_POS_z']].mean(axis=1)
        
        data['l_hip_POS_x'] = data[['l_asis_POS_x', 'l_psis_POS_x']].mean(axis=1)
        data['l_hip_POS_y'] = data[['l_asis_POS_y', 'l_psis_POS_y']].mean(axis=1)
        data['l_hip_POS_z'] = data[['l_asis_POS_z', 'l_psis_POS_z']].mean(axis=1)


        # Drop original columns
        data = data.drop(columns=data.filter(regex=r'(l_th|r_th|l_sk|r_sk)\d+_POS_(x|y|z)'))
        data = data.drop(columns=data.filter(regex='_hd_POS_[xyz]').columns)
        data = data.drop(columns=data.filter(regex='sis_POS_[xyz]').columns)
        data = data.drop(columns=data.filter(regex=r'(m_ster)\d+_POS_(x|y|z)'))
        data = data.drop(columns=data.filter(regex='_err').columns)

    except KeyError as e:
        print(f"Missing key(s) in DataFrame: {e} for subject {sub_id} during {task_name}")
        return None

    
    return data