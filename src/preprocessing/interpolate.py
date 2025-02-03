import kineticstoolkit as ktk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fill_gaps(data, task_name, threshold=200):
    """
    Identify and handle consecutive NaNs in the DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing marker position data.
        threshold (int): Maximum allowable length of consecutive NaNs. Beyond this, data is discarded.
    
    Returns:
        pd.DataFrame: Processed DataFrame with NaNs filled or discarded based on the threshold.
        int: Number of discarded trials due to long NaN sequences.
    """
    discarded_trials = {'walkPreferred' : 0, 
                        'walkFast' : 0, 
                        'walkSlow' : 0}  # Counter for discarded trials

    ts_trimmed = ktk.TimeSeries()
    time = np.array(range(0, len(data['l_toe_POS_x'])))/200
    ts_trimmed.time = time.reshape(time.shape[0])

    for col in data.columns:
        max_nan_streak = 0  # Variable to store the longest NaN sequence
        is_nan = data[col].isna().astype(int)
        nan_streaks = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).cumsum()
        max_streak = nan_streaks.max()  # Get the longest streak in this column

        if max_streak > max_nan_streak:
            max_nan_streak = max_streak  # Update the global max

            if max_nan_streak != 0:
                print(max_nan_streak, col)

        ts_trimmed = ts_trimmed.add_data(col, np.array(data[col]))


    filled_markers = ts_trimmed.fill_missing_samples(max_missing_samples=threshold)


    # fig, ax = plt.subplots()
    # plt.show()
    return data