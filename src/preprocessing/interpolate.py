import kineticstoolkit as ktk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def fill_gaps(data: pd.DataFrame, sub_id: str, task_name: str, fc: float, threshold: float) -> pd.DataFrame:
    """
    Identify and handle consecutive NaNs in the DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing marker position data.
        task_name (str): Name of the task currently analysed.
        threshold (int): Maximum allowable length of consecutive NaNs. Beyond this, data is discarded.
    
    Returns:
        pd.DataFrame: Processed DataFrame with NaNs filled or discarded based on the threshold.
        int: Number of discarded trials due to long NaN sequences.
    """
    discarded_trials = {'walkPreferred' : 0, 
                        'walkFast' : 0, 
                        'walkSlow' : 0}  # Counter for discarded trials

    ts = ktk.TimeSeries()
    time = np.array(range(0, len(data['l_toe_POS_x'])))/200
    ts.time = time.reshape(time.shape[0])

    for col in data.columns:
        ts = ts.add_data(col, np.array(data[col]))

    filled_markers = ts.fill_missing_samples(max_missing_samples=threshold) 

    filled_df = pd.DataFrame(filled_markers.data)

    for col in filled_df:
        max_nan_streak = 0  # Variable to store the longest NaN sequence
        is_nan = filled_df[col].isna().astype(int)
        nan_streaks = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).cumsum()
        max_streak = nan_streaks.max()  # Get the longest streak in this column

        if max_streak > max_nan_streak:
            max_nan_streak = max_streak  # Update the global max

            if max_nan_streak != 0:
                print('\n', task_name)
                print(max_nan_streak, col)

    if pd.DataFrame(filled_markers.data).isnull().values.any():
        print(f'{sub_id} {task_name} has NaNs')

    # filter
    filled_markers = ktk.filters.butter(filled_markers, fc=fc) # default order = 2

    return pd.DataFrame(filled_markers.data) # return data with filled gaps as a dataframe



def recacl_clusters(data: pd.DataFrame, sub_id: str, task_name: str):
    """
    Fills missing marker positions using average distances from the last 5 fully visible frames.

    Args:
        data (pd.DataFrame): Input DataFrame with marker position data.

    Returns:
        pd.DataFrame: Data with missing marker positions reconstructed.
    """

   # Define marker clusters
    clusters = {
        "sternum": ["m_ster1", "m_ster2", "m_ster3"],
        "thigh_r": ["r_th1", "r_th2", "r_th3", "r_th4"],
        "thigh_l": ["l_th1", "l_th2", "l_th3", "l_th4"],
        "shank_r": ["r_sk1", "r_sk2", "r_sk3", "r_sk4"],
        "shank_l": ["l_sk1", "l_sk2", "l_sk3", "l_sk4"]
    }

    filled_data = data.copy()  # Create a copy to modify

    for cluster_name, markers in clusters.items():
        cluster_cols = {axis: [m + f"_POS_{axis}" for m in markers] for axis in ['x', 'y', 'z']}

        # Check if this cluster has missing data
        if not filled_data[sum(cluster_cols.values(), [])].isna().any().any():
            # print(f"No missing markers in {cluster_name}, skipping.")
            continue  # Skip to the next cluster

        # print(f"Processing {cluster_name} (missing markers detected)...")

        # Find last 5 fully visible rows before NaNs start
        valid_rows = filled_data.dropna(subset=sum(cluster_cols.values(), []))
        if valid_rows.empty:
            # print(f"No complete data found for {cluster_name}, skipping.")
            continue  # Skip this cluster if no complete data exists

        last_valid_rows = valid_rows.iloc[-5:]  # Take last 5 valid rows before NaNs start

        # Compute average distances separately for x, y, and z
        avg_distances = {axis: {} for axis in ['x', 'y', 'z']}
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                for axis in ['x', 'y', 'z']:
                    col_i, col_j = markers[i] + f"_POS_{axis}", markers[j] + f"_POS_{axis}"
                    d_values = (last_valid_rows[col_i] - last_valid_rows[col_j]).values
                    avg_distances[axis][(markers[i], markers[j])] = np.mean(d_values)  # Store avg per axis

        # print(f"Computed average distances for {cluster_name} (last 5 frames): {avg_distances}")

        # **Step 2: Fill Missing Values Using Computed Distances**
        for axis in ['x', 'y', 'z']:
            for marker in markers:
                col_marker = marker + f"_POS_{axis}"

                if filled_data[col_marker].isna().any():  # If marker is missing
                    # print(f"Filling missing values for {marker} in {axis}-direction...")

                    for i, row in filled_data.iterrows():
                        if pd.isna(row[col_marker]):  # If NaN detected
                            # Find a reference marker that is present
                            for ref_marker in markers:
                                if ref_marker != marker:
                                    col_ref = ref_marker + f"_POS_{axis}"
                                    if not pd.isna(row[col_ref]):  # Found a visible reference marker
                                        # Compute missing marker position using the known reference and stored distances
                                        distance = avg_distances[axis].get((marker, ref_marker), avg_distances[axis].get((ref_marker, marker), None))

                                        if distance is not None:
                                            filled_data.at[i, col_marker] = row[col_ref] + distance
                                            break  # Stop after filling one valid reference

    return filled_data