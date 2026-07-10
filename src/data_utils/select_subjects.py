import pandas as pd
import numpy as np
from typing import Literal

#used
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
    
    return [f"pp{int(s):>03d}" for s in sub_ids] # pp001 format

#used
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
    # Build pp-format IDs from the demographics id column and match against control_ids
    demo_pp_ids = demographics_df["id"].apply(lambda x: f"pp{int(str(x).split('-')[0]):>03d}")
    all_control_demographics_df = demographics_df[demo_pp_ids.isin(control_ids)].sort_values(by='age', ascending=False)

    # Matching the size of control and diagnosis groups
    matched_control_demographics_df = all_control_demographics_df[:n_subs]

    # A list of control group subject ids
    sub_ids = matched_control_demographics_df['id'].unique().tolist()

    return [f"pp{int(s):>03d}" for s in sub_ids] # pp001 format

    


def age_match_controls(demographics_df, control_ids, treatment_ids):
    """Match controls to disease subjects by minimising age difference.

    Uses greedy 1:1 nearest-neighbour matching without replacement.
    For each disease subject (in random order), finds the closest available
    control by age and assigns them. This minimises the overall age difference
    between groups.

    Parameters
    ----------
    demographics_df : pd.DataFrame
        Demographics table with columns ``id`` and ``age``.
    control_ids : list[str]
        Pool of available control IDs (pp-format).
    treatment_ids : list[str]
        Disease group IDs (pp-format). Sets the target size.

    Returns
    -------
    list[str]
        Age-matched control IDs (same length as treatment_ids, or fewer if
        not enough controls available).
    """
    import numpy as np

    def to_pp(x):
        return f"pp{int(str(x).split('-')[0]):>03d}"

    demo_pp = demographics_df.copy()
    demo_pp['pp_id'] = demo_pp['id'].apply(to_pp)

    # Get ages for both groups
    disease_ages = (demo_pp[demo_pp['pp_id'].isin(treatment_ids)]
                    .set_index('pp_id')['age']
                    .to_dict())
    control_ages = (demo_pp[demo_pp['pp_id'].isin(control_ids)]
                    .set_index('pp_id')['age']
                    .to_dict())

    if not disease_ages or not control_ages:
        return []

    available = dict(control_ages)  # mutable pool
    matched = []

    # Match greedily: for each disease subject find closest control
    for sub_id in treatment_ids:
        if not available:
            break
        d_age = disease_ages.get(sub_id)
        if d_age is None:
            continue
        closest = min(available, key=lambda c: abs(available[c] - d_age))
        matched.append(closest)
        del available[closest]

    return matched