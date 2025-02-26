from src.data_utils import data_loader, select_subjects
from src.preprocessing.filter import (
    butter_lowpass_filter
)
from src.data_utils import groups
# from src.kinectome import calculate_crl_mtrx
from src.preprocessing import preprocessing
from src import kinectome, modularity, fingerprint
from pathlib import Path
import sys
import pandas as pd
import src
import os

if sys.platform == "linux":
    RAW_DATA_PATH = Path(
        "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
    )
    BASE_PATH = Path(
        "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"
    )
elif sys.platform == "win32":
    RAW_DATA_PATH = Path(
        "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    )
    BASE_PATH = Path(
        "Z:\\Keep Control\\Data\\lab dataset"
    )

TASK_NAMES = [
    "walkFast", "walkSlow", "walkPreferred" 
]
TRACKING_SYSTEMS = [
    "omc"
] # add "imu" if needed
RUN = [
    'on', 'off'
        ] # add 'off' if needed 
KINEMATICS = [
       'vel', 'pos', 'acc'
                ] # for calculating kinectomes using position, velocity and acceleration data (what about jerk?)
FS = 200 # sampling rate 
# ordered list of markers
MARKER_LIST = ['head', 'ster', 'l_sho', 'r_sho',  
                'l_elbl', 'r_elbl','l_wrist', 'r_wrist', 'l_hand', 'r_hand', 
                'l_asis', 'l_psis', 'r_asis', 'r_psis', 
                'l_th', 'r_th', 'l_sk', 'r_sk', 
                'l_ank', 'r_ank', 'l_toe', 'r_toe']
DIAGNOSIS = ['diagnosis_parkinson'] # a list of diagnoses of interest

PD_ON = ['pp065', 'pp032'] # a list of sub_ids of PD that were measured in on condition




def main() -> None:
    
    # can be done once since it saves the kinectomes as .npy files in the derived_data 
    kinectome.calculate_all_kinectomes(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, RAW_DATA_PATH, FS, BASE_PATH, MARKER_LIST) 
    
    modularity.modularity_main(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, RAW_DATA_PATH, BASE_PATH, MARKER_LIST)

    

    return


if __name__ == "__main__":
    main()