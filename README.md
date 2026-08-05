# pykinectome

A Python pipeline for computing and analysing **kinectomes** — full-body inter-segmental coordination matrices built from optical motion capture data during walking tasks.

Kinectomes capture the pairwise correlation structure of kinematic signals (position, velocity, or acceleration) across all tracked body segments simultaneously, for each gait cycle. The pipeline supports group comparisons, community detection, centrality analysis, dominant coordination pattern identification, and time-lag analysis.

Based on the Keep Control lab dataset (BIDS-formatted optical motion capture, 200 Hz).

---

## Method references

- **Kinectome concept:** Troisi Lopez et al. (2022). *The kinectome: A comprehensive kinematic map of human motion in health and disease.* Annals of the New York Academy of Sciences. 
- **Gait event detection:** Bonci et al. (2022). *An algorithm for accurate marker-based gait event detection in healthy and pathological populations during complex motor tasks.* Frontiers in Bioengineering and Biotechnology.
- **Reference communities:** Kluge et al. (2021); Warmerdam et al. (2021); Meyns et al. (2013).

---

## Requirements

- Python ≥ 3.10
- Dependencies managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Key packages: `numpy`, `scipy`, `pandas`, `networkx`, `python-louvain`, `matplotlib`, `seaborn`, `statsmodels`, `kineticstoolkit`, `scikit-learn`.

---

## Quick start

1. **Edit `config.py`** — this is the only file you need to touch. Set your paths, recording parameters, marker lists, and analysis settings.

2. **Run the pipeline:**

```bash
uv run python main.py
```

All output folders are created automatically under `RESULT_BASE_PATH`.

---

## Configuration (`config.py`)

All parameters are in one place. The file is organised into five sections:

### 1. Paths

```python
RAW_DATA_PATH = Path("/path/to/rawdata")   # raw BIDS motion TSV files
BASE_PATH     = Path("/path/to/dataset")   # parent of rawdata/ and derived_data/
RESULT_BASE_PATH = Path("results")         # all outputs go here
DEMOGRAPHICS_PATH = BASE_PATH / "demographics_scores.xlsx"
```

### 2. Recording parameters

```python
FS = 200                        # sampling rate (Hz)
TRACKING_SYSTEMS = ["omc"]      # "omc" and/or "imu"
TASK_NAMES = ["walkStroop"]     # walking tasks to analyse
RUN = ["on"]                    # medication state for PD: "on", "off", or both
KINEMATICS = ["acc"]            # "pos", "vel", or "acc"
```

### 3. Marker lists

```python
MARKER_LIST        # markers in raw file order (matched to column names)
MARKER_LIST_AFFECT # markers reordered by affected/less-affected side
```

### 4. Group definition

Two modes — set `GROUP_MODE` to choose:

**`"auto"`** (default): Groups are derived from the demographics spreadsheet at `DEMOGRAPHICS_PATH`.

```python
GROUP_MODE = "auto"
DIAGNOSIS  = ["diagnosis_parkinson"]
PD_ON      = ["pp065", "pp032"]   # PD subjects measured on medication
```

**`"manual"`**: Supply subject ID lists directly — no demographics file needed.

```python
GROUP_MODE    = "manual"
TREATMENT_IDS = ["pp001", "pp003", "pp007"]
CONTROL_IDS   = ["pp002", "pp004", "pp008"]
```

### 5. Analysis settings

```python
FULL         = False     # False = separate AP/ML/V kinectomes (recommended)
CORRELATION  = "pears"  # "pears", "cross", or "dcor"
INTERPOL     = True      # use gap-filled kinectome data
TRIM_MODE    = "cone"    # "cone" = detect walkway gate from cone markers
                         # "none" = use full recording as-is

# Modularity
COMMUNITY_THRESHOLD  = 0.7
CLUSTERING_METHOD    = "louvain"
CONSENSUS_COMMUNITIES = [...]   # literature-based reference communities
```

---

## Pipeline steps (`main.py`)

Steps are run by calling the appropriate function in `main.py`. Steps 2–6 all read kinectomes saved by Step 1, so Step 1 only needs to run once.

| Step | Function | Output folder |
|------|----------|---------------|
| 1 | `kinectome.calculate_all_kinectomes` | `BASE_PATH/derived_data/sub-<id>/kinectomes/` |
| 2 | `kinectome_characteristics.compare_between_groups` | `results/kinectome_characteristics/` |
| 3 | `time_lag.time_lag_main` | `results/time_lag/` |
| 4 | `patterns.patterns_main` | `results/patterns/` |
| 5 | `modularity.modularity_main` | `results/modularity/` |
| 6 | `centrality.centrality_main` | `results/centrality/` |

---

## Output structure

```
RESULT_BASE_PATH/
├── kinectome_characteristics/
│   ├── *.png                        # average and std kinectome plots
│   ├── permutation_*.png            # permutation test histograms
│   └── bootstrapping/
│       └── sample_size_analysis/
├── modularity/
│   ├── allegiance_matrices/         # pickled avg/std allegiance matrices
│   └── *.png                        # community and strength plots
├── patterns/
│   └── *.csv                        # pattern value tables
├── centrality/
│   ├── centrality_data_*.pkl
│   └── csv/
├── time_lag/
└── community_plots/
```

Kinectomes (`.npy` files) are saved into the BIDS `derived_data/` tree:

```
BASE_PATH/derived_data/sub-<id>/kinectomes/
    sub-<id>_task-<task>_tracksys-<sys>_<kin>_kinct<start>-<end>_<corr>[_interpol].npy
```

---

## Using your own data

The pipeline is designed to be adapted for different data formats. Two things need to be configured:

### 1. Data loader

The default loader (`BIDSDataLoader`) navigates BIDS folder structure. To use your own:

1. Subclass `DataLoader` from `src/data_utils/data_loader_interface.py`
2. Implement `load_raw_data(sub_id, task_name, tracksys, run) -> pd.DataFrame`
3. Point `config.py` at your class:

```python
from src.data_utils.my_loader import MyDataLoader
DATA_LOADER_CLASS = MyDataLoader
```

Your `load_raw_data` must return a DataFrame where rows are time samples and columns follow the convention `<marker>_POS_<x|y|z>`. NaN is used for missing samples.

### 2. Trimming

Set `TRIM_MODE` in `config.py`:

- `"cone"` — walkway gate detected from cone markers (requires `start_1/2`, `end_1/2` columns in the raw data).
- `"none"` — no trimming; full recording used as-is.

Custom trimming strategies can be added to `src/preprocessing/trim_data.py`.

### 3. Groups

Set `GROUP_MODE = "manual"` in `config.py` and fill `TREATMENT_IDS` / `CONTROL_IDS` directly — no demographics file required.

---

## Gait event detection

Initial contacts (ICs) are detected automatically from heel marker trajectories using the **Bonci et al. (2022)** algorithm, implemented in `src/data_utils/detect_gait_events.py`.

Detection runs on the **raw data** (before preprocessing) so that the walkway cone marker columns are still present for the gate boundaries. The detected events are passed through the pipeline and used to define overlapping gait cycle windows, each containing one full left and one full right stride.

The old approach (loading pre-computed BIDS event `.tsv` files) is preserved as commented-out code in `src/data_utils/data_loader.py` and `src/kinectome.py` for reference and easy reversion.

---

## Project structure

```
pykinectome/
├── main.py                          # entry point — uncomment steps to run
├── config.py                        # all parameters — edit this only
├── README.md
├── pyproject.toml
└── src/
    ├── kinectome.py                 # kinectome computation
    ├── kinectome_characteristics.py # group comparison + permutation tests
    ├── modularity.py                # community detection + allegiance matrices
    ├── patterns.py                  # dominant coordination patterns
    ├── centrality.py                # weighted degree centrality
    ├── time_lag.py                  # time-lag cross-correlation analysis
    ├── data_utils/
    │   ├── data_loader_interface.py # abstract base class for data loading
    │   ├── bids_data_loader.py      # BIDS-format loader (default)
    │   ├── data_loader.py           # file I/O utilities + event detection
    │   ├── detect_gait_events.py    # Bonci et al. (2022) IC detection
    │   ├── groups.py                # group definition (auto or manual)
    │   ├── select_subjects.py       # demographics-based subject selection
    │   ├── permutation.py           # permutation + bootstrap testing
    │   ├── plotting.py              # all figure generation
    │   └── statistics.py            # statistical tests
    ├── graph_utils/
    │   ├── graphs.py                # graph construction + centrality measures
    │   └── kinectome2pattern.py     # subgraph pattern extraction
    └── preprocessing/
        ├── preprocessing.py         # main preprocessing chain
        ├── trim_data.py             # walking window detection + trimming
        ├── interpolate.py           # gap filling
        ├── align.py                 # PCA-based walking direction alignment
        ├── differentiation.py       # position → velocity → acceleration
        ├── filter.py                # Butterworth low-pass filter
        └── plot_interpolated.py     # gap-filling inspection plots
```

---

## Demographics file format (auto mode)

When `GROUP_MODE = "auto"`, the demographics spreadsheet at `DEMOGRAPHICS_PATH` must contain at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Numeric subject ID (pipeline formats as `pp001`) |
| `age` | float | Age in years (used for control matching) |
| `diagnosis_parkinson` | 0/1 | 1 = has Parkinson's disease |
| `diagnosis_old` | 0/1 | 1 = older healthy control |
| `diagnosis_young` | 0/1 | 1 = younger healthy control |
| `med_state` | str | `'on'` or `'off'` (PD subjects only) |

---

## Author

Karolina Sägner — Neurogeriatrics, UKSH / Kiel University  
`karolina.saegner@uksh.de`
