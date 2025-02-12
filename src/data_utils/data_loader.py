import pandas as pd
from pathlib import Path
import os


def load_file(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df

def load_events(base_path, sub_id, task_name, run):

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