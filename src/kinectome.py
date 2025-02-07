import os
from src.data_utils import data_loader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def find_gait_cycles(data: pd.DataFrame, sub_id: str, task_name: str, run: str):
    """
    reads the events file and returns indices of full left and right gait cycles between start and stop onsets
    
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


def segment_data(data, cycle_indices):

    cycle_data = data[cycle_indices[0]:cycle_indices[1]]

    return cycle_data
    

def calculate_kinectome(data: pd.DataFrame, sub_id: str, task_name: str, run: str, tracksys: str, base_path):

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

        # Define file name
        file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}.npy"
        file_path = os.path.join(kinectome_path, file_name)

        # Save kinectomes (as numpy array)
        np.save(file_path, correlation_matrices)

        


    return kinectome


