import pandas as pd
from src.data_utils import select_subjects


def define_groups(diagnosis):
    """ Makes the groups based on 
    """
    demographics_df = pd.read_excel(
        r"Z:\Keep Control\Data\demographics_scores_internal_use_only.xlsx"
    )

    for diagnosis in diagnosis:
        if diagnosis == 'diagnosis_parkinson':
            run = 'on' # to identify only those PD subjects that were measured on medication

        # select the subjects for analysis based on their diagnosis and find a evenly sized control group 
        disease_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = diagnosis, run=run)
        all_control_sub_ids = select_subjects.select_subjects_ids(demographics_df, diagnosis = ['diagnosis_old', 'diagnosis_young'])
        matched_control_sub_ids = select_subjects.make_control_group(demographics_df, control_ids=all_control_sub_ids, treatment_ids=disease_sub_ids)
    
    return disease_sub_ids, matched_control_sub_ids