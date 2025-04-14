from src.data_utils import data_loader, select_subjects
from src.preprocessing.filter import (
    butter_lowpass_filter
)
from src.data_utils import groups
# from src.kinectome import calculate_crl_mtrx
from src.preprocessing import preprocessing
from src import kinectome, modularity, kinectome_characteristics
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
    "walkPreferred", "walkFast", "walkSlow", 
]
TRACKING_SYSTEMS = [
    "omc"
] # add "imu" if needed
RUN = [
    'on'
        ] # add 'off' if needed 
KINEMATICS = [
       'acc'
                ] # for calculating kinectomes using position, velocity and acceleration data (what about jerk?)
FS = 200 # sampling rate 
# ordered list of markers
MARKER_LIST = ['head', 'ster', 'l_sho', 'r_sho',  
                'l_elbl', 'r_elbl','l_wrist', 'r_wrist', 'l_hand', 'r_hand', 
                'l_asis', 'l_psis', 'r_asis', 'r_psis', 
                'l_th', 'r_th', 'l_sk', 'r_sk', 
                'l_ank', 'r_ank', 'l_toe', 'r_toe']

MARKER_LIST_AFFECT = [
        'head', 'ster',
        'sho_la', 'sho_ma', 'elbl_la', 'elbl_ma', 'wrist_la', 'wrist_ma', 'hand_la', 'hand_ma',
        'asis_la', 'asis_ma', 'psis_la', 'psis_ma',
        'th_la', 'th_ma', 'sk_la', 'sk_ma',
        'ank_la', 'ank_ma', 'toe_la', 'toe_ma'
    ] # desired order of markers after sorting more and less affected sides 
    
DIAGNOSIS = ['diagnosis_parkinson'] # a list of diagnoses of interest

FULL = False # True or False depending if full kinectome (all three directions in one kinectome) should be analysed

CORRELATION = 'pears' # 'pears', 'cross' or 'dcor' are the options, depending on which correlation methods should be used for building the kinectomes

PD_ON = ['pp065', 'pp032'] # a list of sub_ids of PD that were measured in on condition

# path where the results of modularity analysis (std within subjects (csv), avg subject allegiance matrices (pkl) are stored)
RESULT_BASE_PATH = 'C:/Users/Karolina/Desktop/pykinectome/results'

def main() -> None:
    
    # can be done once since it saves the kinectomes as .npy files in the derived_data 
    # kinectome.calculate_all_kinectomes(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, RAW_DATA_PATH, FS, BASE_PATH, MARKER_LIST, RESULT_BASE_PATH, FULL) 
    
    # investigate kinectome characteristics (mean and standard deviation of the kinectomes)
    # uses permutation analysis (Spearman's rho) to check if the matrices correlate with one another 
    kinectome_characteristics.compare_between_groups(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, BASE_PATH, MARKER_LIST_AFFECT, RESULT_BASE_PATH, FULL, CORRELATION)



    modularity.modularity_main(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, BASE_PATH, MARKER_LIST, RESULT_BASE_PATH, FULL, CORRELATION)
                            
                            #    analyse_intra_sub = False, visualise=False, 
                               
                            #    avg_per_speed_computed = False, result_path = RESULT_BASE_PATH)
    # centrality.centrality_main(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON, BASE_PATH, MARKER_LIST, RESULT_BASE_PATH)

    

    return


if __name__ == "__main__":
    main()