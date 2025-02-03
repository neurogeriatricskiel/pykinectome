import kineticstoolkit as ktk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def fill_gaps(data, task_name, fc, threshold):
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
        # max_nan_streak = 0  # Variable to store the longest NaN sequence
        # is_nan = data[col].isna().astype(int)
        # nan_streaks = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).cumsum()
        # max_streak = nan_streaks.max()  # Get the longest streak in this column

        # if max_streak > max_nan_streak:
        #     max_nan_streak = max_streak  # Update the global max

        #     if max_nan_streak != 0:
        #         print('\n', task_name)
        #         print(max_nan_streak, col)

        ts = ts.add_data(col, np.array(data[col]))

    # make an if statement(s) that if any marker in the cluster, or e.g., radial side of the wrist is missing (while ulnar is visible), then the threshold should not exist
    # or is this made automatically?  
    filled_markers = ts.fill_missing_samples(max_missing_samples=threshold) 

    # filter
    filled_markers = ktk.filters.butter(filled_markers, fc=fc) # default order = 2

    return pd.DataFrame(filled_markers.data) # return data with filled gaps as a dataframe