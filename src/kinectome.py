import os
from src.data_utils import data_loader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def find_gait_cycles(base_path, data: pd.DataFrame, sub_id: str, task_name: str, run: str):
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
    

def calculate_kinectome(data: pd.DataFrame, sub_id: str, task_name: str, run: str, tracksys: str, kinematics: str, base_path, marker_list):
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

    gait_cycles, start_onset = find_gait_cycles(base_path, data, sub_id, task_name, run)

    for i in range(len(gait_cycles)):
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

        # Initialize correlation matrices
        num_markers = len(marker_names)
        correlation_matrices = np.zeros((num_markers, num_markers, 3))

        # Compute correlation for each coordinate (x, y, z)
        for i, coord in enumerate([f'_{kinematics.upper()}_x', f'_{kinematics.upper()}_y', f'_{kinematics.upper()}_z']):
            markers = [m + coord for m in marker_list]
            correlation_matrices[:, :, i] = gait_cycle_data[markers].corr(method='pearson', min_periods=1)
     
        # directory to save 
        kinectome_path = f"{base_path}\\derived_data\\sub-{sub_id}\\kinectomes"

        # Ensure directory exists
        if not os.path.exists(kinectome_path):
            os.makedirs(kinectome_path)

        # Define file name (_pos_ for kinetomes of marker position data, vel - velocity, acc - acceleration)
        if run: # 'run-off' or 'run-on' will appear in the kinectome file name
            file_name = f"sub-{sub_id}_task-{task_name}_run-{run}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}.npy"
        else: 
            file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_{kinematics}_kinct{cycle_indices[0]+start_onset}-{cycle_indices[1]+start_onset}.npy"
        
        file_path = os.path.join(kinectome_path, file_name)

        # visualise_kinectome(correlation_matrices, 'test_plot_kinectome_pres.png', marker_list, sub_id, task_name, kinematics)
        
        # Save kinectomes (as numpy array)
        np.save(file_path, correlation_matrices)   
        

def visualise_kinectome(kinectome, figname, marker_list, sub_id, task_name, kinematics):
    """
    Plots the kinectomes in AP, ML, and V directions with marker names as labels.
    """

    # Split left and right markers
    left_markers = [m for m in marker_list if m.startswith('l_')]
    right_markers = [m for m in marker_list if m.startswith('r_')]
    middle_markers = ['head', 'ster']
    
    # Combine them in the desired order (left-side first, then right-side)
    ordered_marker_list = middle_markers + left_markers + right_markers

    plt.figure(figsize=(15, 5))
    # Reorder the kinectome numpy array accordingly
    marker_indices = [marker_list.index(m) for m in ordered_marker_list]
    reordered_kinectome = kinectome[np.ix_(marker_indices, marker_indices, [0, 1, 2])]

    # Define vmin/vmax for each direction
    scales = [(0.5, 1) if kinematics == 'pos' else (-1, 1), (-1, 1), (-1, 1)]

    for i, matrix in enumerate(reordered_kinectome.transpose(2, 0, 1)):  # Iterate over 3 matrices
        plt.subplot(1, 3, i + 1)  # Create subplot
        sns.heatmap(matrix, cmap="coolwarm", vmin=scales[i][0], vmax=scales[i][1], square=True, cbar=True,
                    xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)  # Add labels
        
        plt.title(f"Correlation Matrix {['Anteroposterior', 'Mediolateral', 'Vertical'][i]}")
    
    plt.suptitle(f'{kinematics.upper()} kinectomes of {sub_id} during {task_name}')
    os.chdir('C:/Users/Karolina/Desktop/pykinectome/pykinectome/src/preprocessing')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(figname, dpi=600)
