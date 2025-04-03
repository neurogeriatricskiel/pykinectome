import pandas as pd
from pathlib import Path
import os
import re
import numpy as np


def load_file(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df

def load_events(base_path, sub_id, task_name, run,linux=False):

    """
    Load event data for a given subject, task, and run.

    This function navigates to the motion data directory for a specified subject 
    and identifies event files associated with the given task. If multiple runs 
    ('on' or 'off') exist, it selects the appropriate run-specific file. If no 
    run-specific files exist, it selects a general event file.

    Parameters:
    ----------
    base_path : str
        The base directory containing raw data.
    sub_id : str
        The subject identifier.
    task_name : str
        The name of the task to filter event files.
    run : str
        The specific run condition ('on' or 'off').
    linux : bool, optional
        Flag to indicate if the function is running on a Linux system.

    Returns:
    -------
    events : dict or DataFrame
        The loaded event data from the selected file.

    Notes:
    ------
    - Assumes event files are in TSV format.
    - Uses `load_file()` to read the selected event file.
    - If no matching file is found, the function may raise an IndexError.

    """
    if linux:
        os.chdir(f'{base_path}/rawdata/sub-{sub_id}/motion')
    else:
        os.chdir(f'{base_path}\\rawdata\\sub-{sub_id}\\motion')
    file_list = os.listdir()
    event_files = [file for file in file_list if task_name in file and 'events' in file]

     # Filter event files based on 'run' condition
    if any(f"run-{r}" in file for r in ['on', 'off'] for file in event_files):
        event_files = [file for file in event_files if f"run-{run}" in file and '.tsv' in file]
    else:
        event_files = [file for file in event_files if not any(f"run-{r}" in file for r in ['on', 'off']) and '.tsv' in file]

    events = load_file(event_files[0])

    return events

def load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method):
    """Loads kinectome files and sorts them by onset indices."""
    try:
        os.chdir(f'{base_path}/derived_data/sub-{sub_id}/kinectomes')
        file_list = os.listdir()
        
        # run 'on' or 'off' only exists in the file names of pwPD 
        if run:            
            if full:
                relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, run, kinematics, correlation_method, 'full'])]
            else:
                relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, run, kinematics, correlation_method]) and 'full' not in file]
        else:            
            if full:
                relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, kinematics, correlation_method, 'full'])]
            else:
                relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, kinematics, correlation_method]) and 'full' not in file]
        
        sorted_files = sorted(relevant_files, key=lambda file: extract_onset_indices(file)[0])
        
        if not sorted_files:
            return None
        
        return [np.load(file) for file in sorted_files]

    except FileNotFoundError:
        return None

def extract_onset_indices(filename):
    """Extract numerical onset indices from the kinectome filename.
    Part of load_kinectome function. 
    """

    match = re.search(r'kinct(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


def merge_dicts(list_of_dicts):
    import collections 
    result = collections.defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            result[key].append(value)
    return result