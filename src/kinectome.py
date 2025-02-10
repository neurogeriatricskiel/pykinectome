import os
from src.data_utils import data_loader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def find_gait_cycles(data: pd.DataFrame, sub_id: str, task_name: str, run: str):
    """
    Identifies full left and right gait cycles based on event markers within the valid time range.

    Parameters:
    - data (pd.DataFrame): The motion tracking data.
    - sub_id (str): Subject identifier.
    - task_name (str): Name of the task performed.
    - run (str): Specifies the run condition (e.g., "on" or "off" for PD subjects).

    Returns:
    - gait_cycles (list of tuples): List of (start, end) indices for each detected gait cycle.
    - start_onset (int): The index corresponding to the start of the valid motion tracking period.
    """

    file_list = os.listdir()
    event_files = [file for file in file_list if task_name in file and 'events' in file]

     # Filter event files based on 'run' condition
    if any(f"run-{r}" in file for r in ['on', 'off'] for file in event_files):
        event_files = [file for file in event_files if f"run-{run}" in file and '.tsv' in file]
    else:
        event_files = [file for file in event_files if not any(f"run-{r}" in file for r in ['on', 'off']) and '.tsv' in file]

    
    events = data_loader.load_file(event_files[0])

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
    

def calculate_kinectome(data: pd.DataFrame, sub_id: str, task_name: str, run: str, tracksys: str, kinematics: str, base_path):
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

    Returns:
    - None: The function saves correlation matrices but does not return a value.
    """

    gait_cycles, start_onset = find_gait_cycles(data, sub_id, task_name, run)

    for i in range(len(gait_cycles)):
        cycle_indices = gait_cycles[i]

        gait_cycle_data = segment_data(data, cycle_indices)

        # Extract marker names
        marker_names = sorted(set(col[:-6] for col in gait_cycle_data.columns if col.endswith('_POS_x')))
        
        # Initialize correlation matrices
        num_markers = len(marker_names)
        correlation_matrices = np.zeros((num_markers, num_markers, 3))

        # Compute correlation for each coordinate (x, y, z)
        for i, coord in enumerate(['_POS_x', '_POS_y', '_POS_z']):
            markers = [m + coord for m in marker_names]
            correlation_matrices[:, :, i] = gait_cycle_data[markers].corr(method='pearson', min_periods=1)

        # directory to save 
        kinectome_path = f"{base_path}\\derived_data\\sub-{sub_id}\\kinectomes"

        # Ensure directory exists
        if not os.path.exists(kinectome_path):
            os.makedirs(kinectome_path)

        # Define file name (_pos_ for kinetomes of marker position data, vel - velocity, acc - acceleration)
        file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}.npy"
        file_path = os.path.join(kinectome_path, file_name)

        # Save kinectomes (as numpy array)
        np.save(file_path, correlation_matrices)        
