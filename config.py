"""
config.py — Central configuration for the pykinectome analysis pipeline.
=========================================================================

This is the **only file you need to edit** before running the pipeline.

Set the paths and parameters below to match your data and study design,
then run ``python main.py``.  All output folders are created automatically
under ``RESULT_BASE_PATH``.

Sections
--------
1. Paths
2. Recording parameters
3. Marker lists
4. Study design
5. Analysis settings
"""

from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================

# Root folder of your raw data (BIDS-formatted: contains sub-<id>/motion/...)
# Set BOTH paths for your machine and comment out the other.
#
# Work laptop (server):
# RAW_DATA_PATH = Path("/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata")
# BASE_PATH     = Path("/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset")
#
# Work laptop (Windows, Z: drive):
# RAW_DATA_PATH = Path("Z:\\Keep Control\\Data\\lab dataset\\rawdata")
# BASE_PATH     = Path("Z:\\Keep Control\\Data\\lab dataset")
#
# Personal laptop (Windows, test data):
RAW_DATA_PATH = Path("C:\\Users\\Karolina\\Desktop\\dual\\data")
BASE_PATH     = Path("C:\\Users\\Karolina\\Desktop\\dual")

# Path to the demographics/scores spreadsheet (.xlsx).
# This file must contain columns for subject ID, diagnosis flags, age, and
# (for PD subjects) medication state.  See README.md for the expected format.
DEMOGRAPHICS_PATH = BASE_PATH / "demographics_scores_internal_use_only.xlsx"

# Optional: local folder to save a copy of every kinectome as it is computed.
# Useful when raw data is on a server — kinectomes are saved there as normal,
# and an additional copy lands here for offline analysis on your laptop.
# Set to None to disable the local copy.
KINECTOME_SAVE_PATH = Path("C:\\Users\\Karolina\\Desktop\\pykinectome\\kinectomes")
# KINECTOME_SAVE_PATH = None  # uncomment to disable

# All results and figures are saved here, organised into sub-folders
# automatically created per analysis module.
#
# Output structure created at runtime:
#   RESULT_BASE_PATH/
#   ├── kinectomes/                 ← computed kinectome matrices (.npy)
#   ├── kinectome_characteristics/  ← permutation & bootstrap results
#   ├── modularity/                 ← allegiance matrices & community plots
#   ├── patterns/                   ← pattern CSV files & figures
#   ├── centrality/                 ← centrality pickles & figures
#   └── time_lag/                   ← time-lag matrices & figures
RESULT_BASE_PATH = Path("results")

# =============================================================================
# 2. RECORDING PARAMETERS
# =============================================================================

# Sampling rate of the motion capture system (Hz)
FS = 200

# Tracking system(s) to include.  Add "imu" if IMU data should also be used.
TRACKING_SYSTEMS = ["omc"]

# Walking tasks to analyse.  Uncomment the speeds you need.
TASK_NAMES = [
    # "walkPreferred",
    # "walkFast",
    # "walkSlow",
    "walkStroop",
    # "walkReact",

]

# Medication state(s) for participants with Parkinson's disease.
# Use ["on"], ["off"], or ["on", "off"] to include both conditions.
RUN = ["on"]

# Kinematic signal(s) to use for building kinectomes.
# Options: "pos" (position), "vel" (velocity), "acc" (acceleration).
KINEMATICS = ["acc"]

# =============================================================================
# 3. MARKER LISTS
# =============================================================================

# Ordered list of marker names as they appear in your raw data files.
# The order must match the column order in your motion capture files.
MARKER_LIST = [
    'head', 'ster',
    'l_sho', 'r_sho',
    'l_elbl', 'r_elbl',
    'l_wrist', 'r_wrist',
    'l_hand', 'r_hand',
    'l_asis', 'l_psis', 'r_asis', 'r_psis',
    'l_th', 'r_th',
    'l_sk', 'r_sk',
    'l_ank', 'r_ank',
    'l_toe', 'r_toe',
]

# Marker list reordered by affected/less-affected side (used for group
# comparisons after lateralisation).  "las" = less-affected side,
# "mas" = more-affected side.
MARKER_LIST_AFFECT = [
    'head', 'sternum',
    'shoulder_las', 'shoulder_mas',
    'elbow_las', 'elbow_mas',
    'wrist_las', 'wrist_mas',
    'hand_las', 'hand_mas',
    'asis_las', 'asis_mas',
    'psis_las', 'psis_mas',
    'thigh_las', 'thigh_mas',
    'shank_las', 'shank_mas',
    'ankle_las', 'ankle_mas',
    'toe_las', 'toe_mas',
]


# Markers to exclude from kinectome analysis (by task).
# Kinectomes are always saved with all markers — exclusion is applied at
# analysis time only, so you never need to recompute.
# Set to {} to include all markers.
EXCLUDE_MARKERS_BY_TASK = {
    "walkStroop":   ["elbow_las", "elbow_mas", "wrist_las", "wrist_mas",
                     "hand_las",  "hand_mas"],
    "walkReact":    ["elbow_las", "elbow_mas", "wrist_las", "wrist_mas",
                     "hand_las",  "hand_mas"],
    "walkReaction": ["elbow_las", "elbow_mas", "wrist_las", "wrist_mas",
                     "hand_las",  "hand_mas"],
}

# =============================================================================
# 4. STUDY DESIGN — GROUP DEFINITION
# =============================================================================
#
# There are two ways to define your groups.  Choose ONE by setting
# GROUP_MODE below, then fill in the corresponding block.
#
#   "auto"   — groups are derived automatically from a demographics
#              spreadsheet (the original Keep Control workflow).
#              Requires DEMOGRAPHICS_PATH and DIAGNOSIS to be set.
#
#   "manual" — you supply the subject ID lists directly.
#              Use this if you have a different data structure,
#              no demographics file, or just want full control.
#
GROUP_MODE = "auto"  # "auto" or "manual"

# --- Mode: auto (demographics-file-based) ------------------------------------
#
# Diagnosis column name(s) in your demographics file.
# Each entry must be a binary column in the spreadsheet (1 = has condition).
DIAGNOSIS = ["diagnosis_parkinson"]

# Subject IDs of participants with Parkinson's disease who were measured
# in the medication-ON condition (used to handle mixed on/off datasets).
PD_ON = ["pp065", "pp032"]

# Default more-affected side when UPDRS data is not available.
# Used by reorder_kinectome_by_affected_side() as a fallback.
# "left"  → left side labelled as more-affected (_ma), right as less-affected (_la)
# "right" → right side labelled as more-affected (_ma), left as less-affected (_la)
AFFECTED_SIDE_DEFAULT = "left"

# --- Mode: manual (supply IDs directly) --------------------------------------
#
# Fill these lists when GROUP_MODE = "manual".
# IDs must be strings in the same format as your data folders
# (e.g. "pp001", "HC_01", "sub-01" -- whatever your files use).
# Both lists can be any length; they do not need to be equal in size.
#
# TREATMENT_IDS = ["pp001", "pp003", "pp007"]   # your clinical/treatment group
# CONTROL_IDS   = ["pp002", "pp004", "pp008"]   # your control group

# =============================================================================
# 5. ANALYSIS SETTINGS
# =============================================================================

# Data loader class — controls how raw motion capture files are found and loaded.
#
# The default (BIDSDataLoader) works for BIDS-formatted OMC data with the
# folder structure: rawdata/sub-<id>/motion/<file>.tsv
#
# To use your own data structure:
#   1. Create a subclass of DataLoader in src/data_utils/ (see bids_data_loader.py
#      as a template and data_loader_interface.py for the full contract).
#   2. Implement load_raw_data(sub_id, task_name, tracksys, run) -> pd.DataFrame
#   3. Replace BIDSDataLoader below with your class.
#
# To use a custom data loader, edit the import in src/kinectome.py:
#   from src.data_utils.your_loader import YourDataLoader
#   loader = YourDataLoader(base_path=base_path, raw_data_path=raw_data_path)

# Trimming mode: how to crop raw data to the active walking window.
#
#   "cone"  — detect start/stop automatically from walkway cone markers in
#             the raw data (Bonci et al. 2022).  Use this for OMC data with
#             physical start/end markers.  Cone columns are dropped after use.
#   "none"  — no trimming; the full recording is used as-is.  Use this if
#             your data is already cropped, or you want the whole recording.
#
# To add a custom trimming strategy, implement it in
# src/preprocessing/trim_data.py and add a new mode string here.
TRIM_MODE = "cone"

# Whether to compute one "full" kinectome combining all three movement
# directions (AP + ML + V) into a single matrix.
# False → three separate kinectomes (AP, ML, V) per gait cycle (recommended).
# True  → one combined kinectome per gait cycle.
FULL = False

# Correlation method used to build kinectomes.
# Options:
#   "pears" — Pearson correlation (fast, assumes linearity)
#   "cross" — maximum cross-correlation (captures time-lagged relationships)
#   "dcor"  — distance correlation (captures non-linear dependencies)
CORRELATION = "pears"

# Whether to use gap-filled (interpolated) kinectome data.
# True is recommended; requires the interpolation preprocessing step.
INTERPOL = True

# --- Permutation & bootstrap testing ---

# Whether to run bootstrap-wrapped permutation testing.
# Set to False to skip bootstrapping and run a single permutation test instead
# (much faster — useful once you have confirmed your analysis pipeline works).
RUN_BOOTSTRAP = False

# Number of marker-shuffle permutations for the main permutation test
# (run once on the full group averages).
N_PERMUTATIONS = 10000

# Number of bootstrap iterations (subject resampling).
# Use 100 for quick testing, 1000+ for final analysis.
N_BOOTSTRAPS = 100

# Number of permutations PER bootstrap iteration.
# Fewer than N_PERMUTATIONS is fine here — 500 gives stable p-value estimates
# per iteration while keeping runtime manageable.
# Total iterations = N_BOOTSTRAPS × N_BOOTSTRAP_PERMUTATIONS × n_tasks × n_directions
N_BOOTSTRAP_PERMUTATIONS = 500

# Fraction of subjects sampled per bootstrap iteration (with replacement).
# 0.8 works well for small samples — keeps most subjects while adding variability.
BOOTSTRAP_SUBSET_FRACTION = 0.8

# --- Pattern analysis ---

# Reference group whose average pattern is used as the template.
# "Control"    → use control group patterns; compare both groups against them
# "Parkinson"  → use clinical group patterns (or whatever your group is named)
# The group name must match exactly what the pipeline produces from your
# DIAGNOSIS column (e.g. "Parkinson" or "Control").
PATTERN_REFERENCE_GROUP = "Control"

# Task to use for pattern analysis (must be one of TASK_NAMES).
PATTERN_TASK = "walkStroop"

# Movement direction for pattern analysis: "AP", "ML", or "V".
PATTERN_DIRECTION = "AP"

# Pattern type: how the strongest path is selected.
# "max_weight" — path with the maximum sum of edge weights (default).
# Other types can be added to src/patterns.py and listed here.
PATTERN_TYPE = "max_weight"

# Pattern path length range to search.
# Longer paths take more time but capture more complex coordination chains.
PATTERN_MIN_LENGTH = 2
PATTERN_MAX_LENGTH = 20

# --- Modularity analysis ---

# Allegiance threshold: two body segments are assigned to the same community
# if their allegiance score exceeds this value.
COMMUNITY_THRESHOLD = 0.7

# These are applied to the shifted edge weights (original Pearson + |min|).
# AP direction tends to have stronger correlations (higher weights) than ML/V.
# Use values that span the typical weight range across all directions.
MODULARITY_THRESHOLD_LIST = [0.1, 0.7, 0.9, 1.1, 1.3]

# Resolution values for the modularity calculation
# (nx.community.modularity(G, communities, weight='weight', resolution=...)).
# resolution < 1 favours larger communities; resolution > 1 favours smaller,
# more fine-grained ones. modularity_main() runs the full resolution-dependent
# analysis (community strength metrics, consensus/per-subject modularity, and
# every associated plot) once per value in this list, in a single run.
MODULARITY_RESOLUTION_LIST = [1.0, 1.5, 2.0]



# Number of Louvain iterations per gait cycle for allegiance matrix computation.
# Higher = more stable but slower. 10 is fast for testing, 100 for final analysis.
LOUVAIN_ITERATIONS = 10

# Community detection algorithm.  Currently only "louvain" is supported.
CLUSTERING_METHOD = "louvain"

# Literature-based functional communities of body segments during walking.
# Based on Kluge et al. (2021), Warmerdam et al. (2021), Meyns et al. (2013).
# These are used as the consensus (reference) communities for comparison.
CONSENSUS_COMMUNITIES = [
    {'head', 'sternum', 'shoulder_las', 'shoulder_mas',
     'asis_las', 'asis_mas', 'psis_las', 'psis_mas'},
    {'elbow_las', 'wrist_las', 'hand_las',
     'thigh_mas', 'shank_mas', 'ankle_mas', 'toe_mas'},
    {'elbow_mas', 'wrist_mas', 'hand_mas',
     'thigh_las', 'shank_las', 'ankle_las', 'toe_las'},
]