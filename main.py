from src.data_utils import data_loader
from src.preprocessing.filter import (
    butter_lowpass_filter, butter_highpass_filter
)
from pathlib import Path
import pandas as pd
import src


RAW_DATA_PATH = Path(
    "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
)
TASK_NAMES = [
    "walkPreferred", "walkFast", "walkSlow"
]
TRACKING_SYSTEMS = [
    "omc", "imu"
]



def main() -> None:
    demographics_df = pd.read_excel(
        r"Z:\Keep Control\Data\demographics_scores_internal_use_only.xlsx"
    )
    for row_idx, row in demographics_df[:120].iterrows():
        sub_id = f"pp{row['id']:>03d}"
        
        # 
        for task_name in TASK_NAMES:
            file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"

    sub_id = "pp002"
    task_name = "walkFast"
    tracksys = "omc"
    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"
    file_path = RAW_DATA_PATH / f"sub-{sub_id}" / "motion" / file_name
    data = data_loader.load_file(file_path=file_path)

    
    for file in DERIVED_DATA_PATH:
        # Preprocessing
        preprocessed_data = butter_lowpass_filter(data=data, fs=200., cutoff=5.0)
        
        # Calculate kinectome
        kinectome = src.kinectome.calculate(preprocessed_data)
        
        # Modulatiry anal

    return


if __name__ == "__main__":
    main()