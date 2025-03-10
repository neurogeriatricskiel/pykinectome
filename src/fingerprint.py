import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from src.data_utils.data_loader import load_kinectomes
from pathlib import Path

def fingerprint_analysis(base_path, sub_id, task_name, tracksys, run, kinematics):
    """Main function for modularity analysis across gait cycles and speeds.
    """

    # # Path to kinectomes folder
    # kinectome_dir = Path(base_path)/ "derived_data" / f"sub-{sub_id}" / "kinectomes"

    # # Get all kinectome files for this subject, task, and condition
    # file_pattern = f"sub-{sub_id}_task-{task_name}_"
    # if run:
    #     file_pattern += f"run-{run}_"
    # file_pattern += f"tracksys-{tracksys}_{kinematics}_kinct*.npy"

    # kinectome_files = sorted(kinectome_dir.glob(file_pattern))

    # # Load Kinectomes
    # kinectomes = [np.load(f) for f in kinectome_files]


    kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics)

    if len(kinectomes) < 2:
        print(f"Not enough kinectomes for fingerprint analysis: {sub_id}, {task_name}")
        return None
    
    # Convert to numpy array (Shape: num_cycles x 33 x 33 x 3)
    kinectomes = np.array(kinectomes)

    # Separate AP, ML, V
    ap_kinectomes = kinectomes[..., 0]  # Extract AP
    ml_kinectomes = kinectomes[..., 1]  # Extract ML
    v_kinectomes = kinectomes[..., 2]   # Extract V



    # Compute similarity
    ap_fingerprint = compute_fingerprint_similarity(ap_kinectomes)
    ml_fingerprint = compute_fingerprint_similarity(ml_kinectomes)
    v_fingerprint = compute_fingerprint_similarity(v_kinectomes)


    return 

def compute_fingerprint_similarity(kinectome_array):
    """
    Compute fingerprint similarity (placeholder function).
    Modify with your actual similarity computation.
    """
    num_cycles = kinectome_array.shape[0]
    similarity_matrix = np.zeros((num_cycles, num_cycles))

    for i in range(num_cycles):
        for j in range(num_cycles):
            similarity_matrix[i, j] = np.corrcoef(kinectome_array[i].flatten(), kinectome_array[j].flatten())[0, 1] # Pearson's

    return similarity_matrix