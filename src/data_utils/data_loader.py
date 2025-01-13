import pandas as pd
from pathlib import Path


def load_file(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df
