import pandas as pd
import numpy as np
from typing import Literal


def select_subjects_ids(demographics_df, diagnosis: str | list[str], run: None | Literal["off", "on"] = None) -> list[str]:
    """
    Selects subject IDs from a demographics DataFrame based on diagnosis and medication state.

    Parameters:
    ----------
    demographics_df : pandas.DataFrame
        A DataFrame containing demographic and diagnostic information. 
    
    diagnosis : str or list[str]
        The diagnosis or list of diagnoses to filter by. Each diagnosis should correspond 
        to a column in `demographics_df` where a value of `1` indicates the presence of the condition.
    
    run : None or Literal["off", "on"], optional
        The medication state to filter for. Relevant only when `diagnosis` includes "parkinson".
        If `None`, the medication state is ignored. Defaults to `None`.

    Returns:
    -------
    list[int]
        A list of unique subject IDs matching the specified diagnosis and medication state.
    
    Notes:
    -----
    - When "parkinson" is included in the `diagnosis` parameter and `run` is specified, 
      the function filters for rows where "med_state" matches the specified value.
    - For other diagnoses or when `run` is `None`, the function ignores the "med_state" column.
    """

    diagnosis = [diagnosis] if isinstance(diagnosis, str) else diagnosis
    sub_ids = []
    for d in diagnosis:
        if ("parkinson" in d) and (run is not None):
            sub_ids += demographics_df[(demographics_df[d]==1) & (demographics_df["med_state"]==run)]["id"].unique().tolist()
        else:
            sub_ids += demographics_df[(demographics_df[d]==1)]["id"].unique().tolist()
    
    return [f"pp{s:>03d}" for s in sub_ids] # pp001 format


def make_control_group(demographics_df, control_ids: list[str], treatment_ids: list[str]):
    """
    Selects subject IDs from a demographics DataFrame based on diagnosis and medication state.

    Parameters:
    ----------
    demographics_df : pandas.DataFrame
        A DataFrame containing demographic and diagnostic information. 
    
    control_ids : list[str]
        A list containing the ids of all healthy controls.

    treatment_ids : list[str]
        A list containing the ids of the group with a diagnosis of interest.     

    Returns:
    -------
    list[int]
        A list of unique subject IDs of the control group matching the size of the group (diagnosis) of interest. 

    """
    
    sub_ids = []
    n_subs = len(treatment_ids)

    # All controls sorted from oldest to youngest
    all_control_demographics_df = demographics_df[demographics_df["id"].isin(int(s.lstrip("pp").lstrip("0")) for s in control_ids)].sort_values(by='age', ascending=False)

    # Matching the size of control and diagnosis groups
    matched_control_demographics_df = all_control_demographics_df[:n_subs]

    # A list of control group subject ids
    sub_ids = matched_control_demographics_df['id'].unique().tolist()

    return [f"pp{s:>03d}" for s in sub_ids] # pp001 format

    
