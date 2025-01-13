from src.data_utils import data_loader
from pathlib import Path


RAW_DATA_PATH = Path(
    "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
)


def main() -> None:
    sub_id = "pp002"
    task_name = "walkFast"
    tracksys = "omc"
    file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"
    file_path = RAW_DATA_PATH / f"sub-{sub_id}" / "motion" / file_name
    data = data_loader.load_file(file_path=file_path)
    return


if __name__ == "__main__":
    main()