from src.data_utils import data_loader, select_subjects
from src.preprocessing.filter import (
    butter_lowpass_filter
)
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

PD_ON = ['pp065'] # a list of sub_ids of PD that were measured in on condition

def main() -> None:
    demographics_df = pd.read_excel(
        r"Z:\Keep Control\Data\demographics_scores_internal_use_only.xlsx"
    )

    # select the subjects for analysis based on their diagnosis and find a evenly sized control group 
    pd_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = 'diagnosis_parkinson', run="on")
    all_control_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = ['diagnosis_old', 'diagnosis_young'])
    matched_control_sub_ids = select_subjects.make_control_group(demographics_df, control_ids=all_control_sub_ids, treatment_ids=pd_sub_ids)
    
    # use for debugging particular subjects
    debug_ids = ['pp021']

    # file name is based on task names and tracking systems defined above
    for sub_id in pd_sub_ids + matched_control_sub_ids:
    # for sub_id in debug_ids:
        for kinematics in KINEMATICS:
            # for sub_id in debug_ids: # use this line for inspecting single subjects with known IDs
            for task_name in TASK_NAMES:
                for tracksys in TRACKING_SYSTEMS:
                    for run in RUN:
                        if sub_id in PD_ON: # those sub ids which are measured in 'on' condition but there is no 'run-on' in the filename
                            run = 'on'
                        else:
                            run = run
                           
                            # change current working directory to the folder with motion data
                            os.chdir(f'{RAW_DATA_PATH}\\sub-{sub_id}\\motion')
                            file_list = os.listdir()

                            file_path = None  # Initialize file_path for each loop

                            for file in file_list:
                                if sub_id in file and task_name in file and tracksys in file and 'motion' in file:
                                    # Check if the 'run' condition matches 
                                    if any(f"run-{r}" in file for r in RUN):
                                        file_path = f"{RAW_DATA_PATH}\\sub-{sub_id}\\motion\\{file}"
                                        break # Exit the loop once a matching file is found

                                    # Include files without any 'run-on' or 'run-off' condition (run-on and run-off are only applicable to PD)
                                    elif not any(f"run-{cond}" in file for cond in ["on", "off"]):
                                        file_path = f"{RAW_DATA_PATH}\\sub-{sub_id}\\motion\\{file}"
                                        break
                                    
                            if file_path:
                                # Load the data as a pandas dataframe
                                data = data_loader.load_file(file_path=file_path)

                                # trim, reduce dimensions, interpolate, rotate, differentiate
                                preprocessed_data = preprocessing.all_preprocessing(data, sub_id, task_name, run, tracksys, kinematics, FS)

                                if preprocessed_data is None:
                                    continue                                

                                # Calculate kinectomes (for each gait cycle) and save in derived_data/kinectomes
                                # can be done only once for each derivative (position, velocity, acceleration etc.) and then commented out to save on running time
                                if sub_id in all_control_sub_ids: # pwPD that were measured on medication and have no 'run' in the filename
                                    run = None
                                # when save=True, then saves the kinectome as .npy files and returns the marker labels for that kinectome. when save=False, only returns the marker labels
                                kinectome.calculate_kinectome(preprocessed_data, sub_id, task_name, run, tracksys, kinematics, BASE_PATH, MARKER_LIST)
                        
                            else:
                                print(f"No matching motion file found for sub-{sub_id}, task-{task_name}, tracksys-{tracksys}, run-{run}")
    return



def modularity(sub_id, task_name, tracksys, run, kinematics, MARKER_LIST):
# Modularity analysis
    modularity_results = modularity.modularity_analysis(BASE_PATH, sub_id, task_name, tracksys, run, kinematics, MARKER_LIST)
    return





if __name__ == "__main__":
    main()