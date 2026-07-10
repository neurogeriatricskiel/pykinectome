from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def rotate_data(data: pd.DataFrame, sub_id: str, task_name: str):
    """Rotate marker data so the x-axis aligns with the walking direction.

    Uses PCA on mid-pelvis (x, y) positions to estimate the principal walking
    direction, then applies the resulting rotation matrix to all marker
    position columns.

    NaN frames in the pelvic markers (e.g. from temporary occlusion) are
    excluded from the PCA fit but the rotation is still applied to all frames,
    including those with NaNs.  This is valid because the walking direction is
    constant throughout the trial — it can be reliably estimated from any
    sufficient subset of valid frames.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed marker position data.
    sub_id : str
        Subject identifier (used for logging).
    task_name : str
        Task name (used for logging).

    Returns
    -------
    pd.DataFrame or None
        Rotated marker data, or None if pelvic marker columns are missing
        or if fewer than 2 valid (non-NaN) frames are available for PCA.
    """
    try:
        mid_hip = pd.DataFrame({
            'mid_hip_pos_x': (data['r_asis_POS_x'] + data['l_asis_POS_x'] +
                              data['r_psis_POS_x'] + data['l_psis_POS_x']) / 4,
            'mid_hip_pos_y': (data['r_asis_POS_y'] + data['l_asis_POS_y'] +
                              data['r_psis_POS_y'] + data['l_psis_POS_y']) / 4,
        })
    except KeyError as e:
        print(f"Missing pelvic marker column(s): {e} — sub {sub_id}, task {task_name}. Skipping.")
        return None

    # Drop NaN rows for PCA fitting only — rotation is applied to all frames
    valid = mid_hip.dropna()

    if len(valid) < 2:
        print(f"{sub_id} during {task_name}: fewer than 2 valid pelvic frames "
              f"({len(valid)} available). Cannot estimate walking direction. Skipping.")
        return None

    nan_count = len(mid_hip) - len(valid)
    if nan_count > 0:
        print(f"{sub_id} during {task_name}: {nan_count} NaN frame(s) in pelvic markers "
              f"— excluded from PCA fit, rotation applied to all frames.")

    rotation_matrix = pca(valid)

    rotated_data = data.copy()
    x_cols = [col for col in data.columns if col.endswith('_POS_x')]
    y_cols = [col for col in data.columns if col.endswith('_POS_y')]

    for x_col, y_col in zip(x_cols, y_cols):
        coords = data[[x_col, y_col]].values
        rotated_coords = np.dot(coords, rotation_matrix.T)
        rotated_data[x_col] = rotated_coords[:, 0]
        rotated_data[y_col] = rotated_coords[:, 1]

    return rotated_data


def pca(data: pd.DataFrame):
    """Run PCA and return the rotation matrix.

    Parameters
    ----------
    data : pd.DataFrame
        2-column DataFrame of (x, y) positions with no NaN values.

    Returns
    -------
    np.ndarray
        2x2 rotation matrix (PCA components).
    """
    pca_model = PCA(n_components=2)
    pca_model.fit(X=data)
    return pca_model.components_