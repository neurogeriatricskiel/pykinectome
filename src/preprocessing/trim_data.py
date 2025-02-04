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
        event_files = [file for file in event_files if f"run-{run}" in file]
    else:
        event_files = [file for file in event_files if not any(f"run-{r}" in file for r in ['on', 'off'])]
    
    if not event_files:
        raise FileNotFoundError(f"No events file found for subject {sub_id}, task {task_name}")
    
    events = data_loader.load_file(event_files[0])

    try:
        start_onset = int(events.loc[events['event_type'] == 'start', 'onset'].values[0])
        stop_onset = int(events.loc[events['event_type'] == 'stop', 'onset'].values[0])

    except IndexError:
        raise ValueError(f"Start or stop event missing for subject {sub_id}, task {task_name}")
    
    # Cut the data to be between start and stop
    trimmed_data = data[start_onset:stop_onset]

    return trimmed_data