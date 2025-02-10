import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

def velocity(data: pd.DataFrame, fs: int):
    """
    Computes velocity from position data using finite differences.

    Parameters:
    - data (pd.DataFrame): DataFrame containing position data.
    - fs (int): Sampling frequency in Hz.

    Returns:
    - vel_data (pd.DataFrame): DataFrame containing velocity values for each marker.
    """

    # Calculate the change in position (delta s)
    delta_s = data.diff()

    # Define the time interval (delta t)
    delta_t = 1 / fs

    # Calculate velocity by dividing delta_s by delta_t
    velocity_df = delta_s / delta_t

    # Drop the first row (NaN values due to shifting)
    velocity_df = velocity_df.dropna()

    # Replace '_POS' with '_VEL' in column names
    velocity_df.columns = [col.replace('_POS', '_VEL') for col in velocity_df.columns]

    return velocity_df / 1000 # returns velocity df in m/s (original position data is in mm)


def acceleration(data: pd.DataFrame, fs: int):
    """
    Computes acceleration from position data using finite differences.

    Parameters:
    - data (pd.DataFrame): DataFrame containing position data.
    - fs (int): Sampling frequency in Hz.

    Returns:
    - acceleration_df (pd.DataFrame): DataFrame containing filtered acceleration values for each marker.
    """

    # Compute velocity (already in m/s)
    velocity_df = velocity(data, fs)

    # Define the time interval (delta t)
    delta_t = 1 / fs

    # Compute acceleration as the derivative of velocity
    acceleration_df = velocity_df.diff() / delta_t

    # Drop the first row (NaN values due to shifting)
    acceleration_df = acceleration_df.dropna()

    # Replace '_VEL' with '_ACC' in column names
    acceleration_df.columns = [col.replace('_VEL', '_ACC') for col in acceleration_df.columns]

    # Apply Butterworth low-pass filter (order=2, cutoff=4 Hz)
    cutoff_freq = 4  # Hz
    nyquist = 0.5 * fs
    b, a = butter(N=2, Wn=cutoff_freq / nyquist, btype='low', analog=False)
    
    acceleration_df = acceleration_df.apply(lambda col: filtfilt(b, a, col), axis=0)

    return acceleration_df