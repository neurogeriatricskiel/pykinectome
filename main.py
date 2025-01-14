from src.data_utils import data_loader, select_subjects
from src.preprocessing.filter import (
    butter_lowpass_filter
)
# from src.kinectome import calculate_crl_mtrx
from src.preprocessing import interpolate, align
from pathlib import Path
import sys
import pandas as pd
import src


if sys.platform == "linux":
    RAW_DATA_PATH = Path(
        "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
    )
elif sys.platform == "win32":
    RAW_DATA_PATH = Path(
        "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    )

TASK_NAMES = [
    "walkPreferred", "walkFast", "walkSlow"
]
TRACKING_SYSTEMS = [
    "omc"
] # add "imu" if needed


def main() -> None:
    demographics_df = pd.read_excel(
        r"Z:\Keep Control\Data\demographics_scores_internal_use_only.xlsx"
    )

    # select the subjects for analysis based on their diagnosis and find a evenly sized control group 
    pd_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = 'diagnosis_parkinson', run="on")
    all_control_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = ['diagnosis_old', 'diagnosis_young'])
    matched_control_sub_ids = select_subjects.make_control_group(demographics_df, control_ids=all_control_sub_ids, treatment_ids=pd_sub_ids)
    
    # for row_idx, row in matched_control_sub_ids.iterrows():
        # sub_id = f"pp{row['id']:>03d}"
        
        # file name is based on task names and tracking systems defined above
    for sub_id in matched_control_sub_ids:
        for task_name in TASK_NAMES:
            for tracksys in TRACKING_SYSTEMS:    
                file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"
                file_path = f"{RAW_DATA_PATH}\\sub-{sub_id}\\motion\\{file_name}"

                # Load the data as a pandas dataframe
                data = data_loader.load_file(file_path=file_path)

                # Fill the gaps
                interpolated_data = interpolate(data)

                # Filtering
                preprocessed_data = butter_lowpass_filter(data=interpolated_data, fs=200., cutoff=5.0)
        
                # Calculate kinectome
                kinectome = src.kinectome.calculate(preprocessed_data)
        
                # Modulatiry anal

    return


if __name__ == "__main__":
    main()