from src.preprocessing import align, differentiation, filter, interpolate, trim_data
import os
import numpy as np


def all_preprocessing(data, sub_id, task_name, run, tracksys, kinematics, fs):

    # trim the data to be between the start and finish lines (5m walk)
    trimmed_data = trim_data.startStop(data, sub_id, task_name, run)

    if trimmed_data is None or trimmed_data.empty:
        return None
    
    # long periods of NaNs removed and the data is cut accordingly, returning the cut indices as well
    trimmed_data, nan_idx = trim_data.remove_long_nans(trimmed_data, sub_id, task_name, run)
    
    if trimmed_data is None or trimmed_data.empty:
        return None

    # recalculate the positions of clusters (always at fixed distance between one another)
    # full_cluster_data = interpolate.recacl_clusters(trimmed_data, sub_id, task_name)

    # reduce the data dimensions (cluster markers calculated into one point)
    reduced_data = trim_data.reduce_dimensions_clusters(trimmed_data, sub_id, task_name)

    if reduced_data is None or reduced_data.empty:
        return None

     # Fill the gaps and filter the data (filter function available in kinetics toolkit) for omc data
    if tracksys =='omc':
        interpolated_data = interpolate.fill_gaps(reduced_data, sub_id, task_name, fc=6, threshold=271) # fc = cut-off for the butterworth filter; threshold = maximum allowed data gap

        # Principal component analysis (to align the x axis with walking direction)
        rotated_data = align.rotate_data(interpolated_data, sub_id, task_name)

    if rotated_data is None or rotated_data.empty:
        return None    
    else:
        # differentiate to get velocity or acceleration from position data 
        if kinematics == 'pos':
            diff_data = rotated_data
        elif kinematics == 'vel':
            diff_data = differentiation.velocity(rotated_data, fs)
        elif kinematics == 'acc':
            diff_data = differentiation.acceleration(rotated_data, fs)
    

    return diff_data