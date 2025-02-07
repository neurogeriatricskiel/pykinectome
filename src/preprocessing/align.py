from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Use PCA for aligning marker data with main walking direction

def rotate_data(data: pd.DataFrame, sub_id: str, task_name: str):
    """
    Rotates marker data using PCA based on pelvic marker midpoints.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input marker position data
    sub_id : str
        Subject identifier
    task_name : str
        Name of the task being performed
    
    Returns:
    --------
    pd.DataFrame
        Rotated marker position data
    """
    
    # Calculate mid hip positions
    try:
        mid_hip = {
            'mid_hip_pos_x': (data['r_asis_POS_x'] + data['l_asis_POS_x'] + data['r_psis_POS_x'] + data['l_psis_POS_x'])/4,
            'mid_hip_pos_y': (data['r_asis_POS_y'] + data['l_asis_POS_y'] + data['r_psis_POS_y'] + data['l_psis_POS_y'])/4
        }
    except KeyError as e:
        print(f"Missing key(s) in DataFrame: {e} for subject {sub_id} during {task_name}")
        return None
    
    # Prepare data for PCA (x and y positions)
    pca_data = pd.DataFrame(mid_hip)

    # check if pelvic data contains NaNs (if so, PCA is impossible; therefore skip this task)
    if pca_data.isna().any().any():
        print(f'{sub_id} during {task_name} has NaN values in the pelvis marker data. Unable to run PCA. Skipping...')
        return None
    
    else:
        # Get rotation matrix
        rotation_matrix = pca(pca_data)

        # Prepare rotated dataframe
        rotated_data = data.copy()

        # Find all x and y position columns
        x_cols = [col for col in data.columns if col.endswith('_POS_x')]
        y_cols = [col for col in data.columns if col.endswith('_POS_y')]

        # Rotate x and y positions
        for x_col, y_col in zip(x_cols, y_cols):
            # Create coordinate matrix
            coords = data[[x_col, y_col]].values
            
            # Rotate coordinates
            rotated_coords = np.dot(coords, rotation_matrix.T)
            
            # Update dataframe
            rotated_data[x_col] = rotated_coords[:, 0]
            rotated_data[y_col] = rotated_coords[:, 1]
    
    return rotated_data


def pca(data: pd.DataFrame):
    ''' Runs principal component analysis.
    Returns a rotation matrix'''
    # apply principal component analysis
    pca = PCA(n_components=2)
    pca.fit(X=data)

    # get the rotation matrix
    rotation_matrix = pca.components_

    return rotation_matrix