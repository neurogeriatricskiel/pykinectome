"""
main.py — Entry point for the pykinectome analysis pipeline.
=============================================================

All parameters are set in ``config.py``.  Edit that file first, then run:

    python main.py

Pipeline steps (uncomment the ones you want to run)
-----------------------------------------------------
1. calculate_all_kinectomes   — builds and saves kinectome matrices per subject
2. compare_between_groups     — kinectome characteristics + permutation tests
3. time_lag_main              — time-lag analysis between groups
4. patterns_main              — dominant coordination pattern analysis
5. modularity_main            — community detection + allegiance matrices
6. centrality_main            — weighted degree centrality per community

Steps 2-6 all read the kinectomes saved by step 1, so step 1 only needs to
be run once.  After that you can run any combination of steps 2-6 independently.
"""

from src import (
    kinectome,
    kinectome_characteristics,
    time_lag,
    patterns,
    modularity,
    centrality,
)

from src.data_utils.demographics_statistics import compare_group_demographics

from config import (
    RAW_DATA_PATH,
    BASE_PATH,
    RESULT_BASE_PATH,
    FS,
    TRACKING_SYSTEMS,
    TASK_NAMES,
    RUN,
    KINEMATICS,
    MARKER_LIST,
    MARKER_LIST_AFFECT,
    DIAGNOSIS,
    PD_ON,
    FULL,
    CORRELATION,
    INTERPOL,
    COMMUNITY_THRESHOLD,
    CLUSTERING_METHOD,
    CONSENSUS_COMMUNITIES,
    PATTERN_REFERENCE_GROUP,
    PATTERN_TASK,
    PATTERN_DIRECTION,
    PATTERN_MIN_LENGTH,
    PATTERN_MAX_LENGTH,
)


def main() -> None:

    # -------------------------------------------------------------------------
    # Step 1 — Build and save kinectomes
    # Run this once. Kinectomes are saved as .npy files under
    # BASE_PATH/derived_data/sub-<id>/kinectomes/
    # -------------------------------------------------------------------------
    # kinectome.calculate_all_kinectomes(
    #     DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON,
    #     RAW_DATA_PATH, FS, BASE_PATH, MARKER_LIST, RESULT_BASE_PATH,
    #     FULL, CORRELATION, INTERPOL,
    # )

    # -------------------------------------------------------------------------
    # Step 2 — Kinectome characteristics
    # Computes mean and standard deviation matrices per group, then uses
    # permutation testing (Spearman's rho) and bootstrapping to test whether
    # matrices differ between groups.
    # -------------------------------------------------------------------------

    # compare_group_demographics(DIAGNOSIS, TASK_NAMES, RESULT_BASE_PATH)

    # kinectome_characteristics.compare_between_groups(
    #     DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON,
    #     BASE_PATH, MARKER_LIST_AFFECT, RESULT_BASE_PATH, FULL, CORRELATION, INTERPOL,
    # )

    # -------------------------------------------------------------------------
    # Step 3 — Time-lag analysis (optional)
    # Tests whether one body segment systematically leads or lags another
    # across the two groups.
    # -------------------------------------------------------------------------
    # time_lag.time_lag_main(
    #     DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON,
    #     BASE_PATH, MARKER_LIST_AFFECT, RESULT_BASE_PATH, FULL,
    # )

    # -------------------------------------------------------------------------
    # Step 4 — Dominant coordination patterns (optional)
    # Identifies the most consistent inter-segment coordination patterns
    # and compares their strength between groups.
    # Configure PATTERN_REFERENCE_GROUP, PATTERN_TASK, PATTERN_DIRECTION,
    # PATTERN_MIN_LENGTH, PATTERN_MAX_LENGTH in config.py.
    # The output pickle filename is generated automatically — no manual naming needed.
    # -------------------------------------------------------------------------
    # patterns.patterns_main(
    #     MARKER_LIST_AFFECT, DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS,
    #     RUN, PD_ON, BASE_PATH, RESULT_BASE_PATH, FULL, CORRELATION, INTERPOL,
    # )

    # print()
    # -------------------------------------------------------------------------
    # Step 5 — Modularity analysis (optional)
    # Detects functional communities via Louvain clustering on allegiance
    # matrices and compares community structure between groups.
    # -------------------------------------------------------------------------
    # modularity.modularity_main(
    #     DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON,
    #     BASE_PATH, MARKER_LIST_AFFECT, RESULT_BASE_PATH, FULL, CORRELATION,
    #     COMMUNITY_THRESHOLD, CLUSTERING_METHOD, CONSENSUS_COMMUNITIES,
    # )



    # -------------------------------------------------------------------------
    # Step 6 — Centrality analysis (optional)
    # Computes weighted degree centrality per body segment and per community,
    # and tests for group differences.
    # -------------------------------------------------------------------------
    centrality.centrality_main(
        DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, PD_ON,
        BASE_PATH, MARKER_LIST_AFFECT, RESULT_BASE_PATH, FULL,
        CORRELATION, INTERPOL, CONSENSUS_COMMUNITIES,
    )

    print()

if __name__ == "__main__":
    main()