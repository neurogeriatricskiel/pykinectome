import pandas as pd
from src.data_utils import select_subjects


def define_groups(diagnosis):
    """Return subject ID lists for the treatment group and matched controls.

    Behaviour is controlled by ``GROUP_MODE`` in ``config.py``:

    ``"auto"``
        Groups are derived from a demographics spreadsheet at
        ``DEMOGRAPHICS_PATH``.  Participants are selected by diagnosis column,
        and a size-matched control group is built by age (oldest first).
        This is the original Keep Control workflow.

    ``"manual"``
        Subject IDs are read directly from ``TREATMENT_IDS`` and
        ``CONTROL_IDS`` in ``config.py``.  Use this when you have a different
        data structure or no demographics file.

    Parameters
    ----------
    diagnosis : list[str]
        List of diagnosis column names (e.g. ``['diagnosis_parkinson']``).
        Only used in ``"auto"`` mode.

    Returns
    -------
    treatment_ids : list[str]
        Subject IDs of the treatment/clinical group.
    control_ids : list[str]
        Subject IDs of the control group.

    Raises
    ------
    ValueError
        If ``GROUP_MODE`` is not ``"auto"`` or ``"manual"``.
    """
    from config import GROUP_MODE, DEMOGRAPHICS_PATH, DIAGNOSIS
    if GROUP_MODE == "auto":
        return _define_groups_from_demographics(diagnosis)
    elif GROUP_MODE == "manual":
        return _define_groups_manual()
    else:
        raise ValueError(
            f"GROUP_MODE in config.py must be 'auto' or 'manual', got '{GROUP_MODE}'."
        )


def _define_groups_from_demographics(diagnosis):
    """Derive groups from the demographics spreadsheet (auto mode).

    Reads the Excel file at ``DEMOGRAPHICS_PATH``, selects subjects matching
    the given diagnosis, and builds a size-matched control group.

    Parameters
    ----------
    diagnosis : list[str]
        Diagnosis column names to filter on.

    Returns
    -------
    disease_sub_ids : list[str]
    matched_control_sub_ids : list[str]
    """
    from config import DEMOGRAPHICS_PATH
    if DEMOGRAPHICS_PATH.suffix.lower() == '.csv':
        demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
    else:
        demographics_df = pd.read_excel(DEMOGRAPHICS_PATH)

    for diag in diagnosis:
        run = 'on' if diag == 'diagnosis_parkinson' else None

        disease_sub_ids = select_subjects.select_subjects_ids(
            demographics_df, diagnosis=diag, run=run
        )
        all_control_sub_ids = select_subjects.select_subjects_ids(
            demographics_df, diagnosis=['diagnosis_old', 'diagnosis_young']
        )
        matched_control_sub_ids = select_subjects.make_control_group(
            demographics_df,
            control_ids=all_control_sub_ids,
            treatment_ids=disease_sub_ids,
        )

    return disease_sub_ids, matched_control_sub_ids


def _define_groups_manual():
    """Return groups from manually specified ID lists in config (manual mode).

    Returns
    -------
    treatment_ids : list[str]
    control_ids : list[str]

    Raises
    ------
    ImportError
        If ``TREATMENT_IDS`` or ``CONTROL_IDS`` are not defined in config.
    """
    try:
        from config import TREATMENT_IDS, CONTROL_IDS
    except ImportError:
        raise ImportError(
            "GROUP_MODE is 'manual' but TREATMENT_IDS and/or CONTROL_IDS are "
            "not defined in config.py.  Please uncomment and fill in those lists."
        )
    return TREATMENT_IDS, CONTROL_IDS


def get_matched_groups_for_task(diagnosis, task_names):
    """Return age-matched subject IDs filtered to those with kinectomes for each task.

    This is the single entry point for group matching used by ALL analysis
    modules. Call this instead of ``define_groups()`` whenever a per-task
    matched group is needed.

    For each task, both groups are filtered to subjects who have kinectome
    files saved in ``KINECTOME_SAVE_PATH``. Controls are age-matched to the
    disease group size (oldest first). If fewer controls have data, the
    broader unmatched pool is used to top up.

    Parameters
    ----------
    diagnosis : list[str]
        Diagnosis column name(s) from the demographics file.
    task_names : list[str]
        Tasks to build matched groups for.

    Returns
    -------
    task_disease_ids : dict[str, list[str]]
        ``{task_name: [sub_id, ...]}``.
    task_control_ids : dict[str, list[str]]
        ``{task_name: [sub_id, ...]}``.
    """
    from pathlib import Path
    from config import KINECTOME_SAVE_PATH, DEMOGRAPHICS_PATH

    disease_sub_ids, matched_control_sub_ids = define_groups(diagnosis)

    def has_kinectomes(sub_id, task):
        d = Path(KINECTOME_SAVE_PATH) / f"sub-{sub_id}"
        if not d.exists():
            return False
        return any(task in f.name for f in d.iterdir() if f.suffix == '.npy')

    import pandas as pd
    from src.data_utils import select_subjects as _ss
    demo = (pd.read_csv(DEMOGRAPHICS_PATH)
            if DEMOGRAPHICS_PATH.suffix.lower() == '.csv'
            else pd.read_excel(DEMOGRAPHICS_PATH))

    task_disease_ids = {}
    task_control_ids = {}

    for task in task_names:
        avail_disease = [s for s in disease_sub_ids if has_kinectomes(s, task)]

        # Build the pool of all controls who have kinectomes for this task,
        # drawing from the full control population (not just the 29 pre-matched).
        all_ctrl = _ss.select_subjects_ids(
            demo, diagnosis=['diagnosis_old', 'diagnosis_young']
        )
        avail_controls_pool = [s for s in all_ctrl if has_kinectomes(s, task)]

        # Age-match: greedy nearest-neighbour matching by age.
        # For each disease subject, finds the closest available control
        # by age (without replacement). This minimises the overall age
        # difference between groups.
        matched = _ss.age_match_controls(
            demo,
            control_ids=avail_controls_pool,
            treatment_ids=avail_disease,
        )

        task_disease_ids[task] = avail_disease
        task_control_ids[task] = matched

        print(f"  Task {task}: disease n={len(avail_disease)}, "
              f"control n={len(matched)} (age-matched from {len(avail_controls_pool)} available)")

    return task_disease_ids, task_control_ids