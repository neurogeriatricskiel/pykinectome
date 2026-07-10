"""
preprocessing.py — Main preprocessing pipeline for motion capture data.
========================================================================

Applies the full preprocessing chain to raw motion capture data for one
subject/task/run, returning data ready for kinectome computation.

The trimming step (cropping to the active walking window) supports two modes
controlled by ``config.TRIM_MODE``:

``"cone"`` (default for BIDS/OMC data)
    The active window is detected automatically from the walkway cone markers
    in the raw data using the Bonci et al. (2022) algorithm.  The cone marker
    columns are then dropped.  Use this when your data contains physical
    start/end markers that define the valid gait period.

``"none"``
    No trimming is applied.  The full recording is used as-is.  Use this
    when your data is already cropped to the walking period, or when you
    want to analyse the entire recording.

To add a custom trimming strategy (e.g. based on force plates or a trigger
channel), implement it in ``trim_data.py`` and add a new mode string here.
"""

from src.preprocessing import align, differentiation, filter, interpolate, trim_data
from src.data_utils.detect_gait_events import detect_events_from_dataframe
import numpy as np


def all_preprocessing(data, sub_id, task_name, run, tracksys, kinematics, fs,
                      frame_start=None, frame_end=None):
    """Apply the full preprocessing chain to raw motion capture data.

    Steps applied in order:
    1. Trim to the active walking window (mode controlled by ``TRIM_MODE``).
    2. Remove long NaN streaks at the boundaries.
    3. Reduce marker cluster dimensions (multi-marker clusters → single point).
    4. Gap-fill and filter (OMC only).
    5. Rotate data so the x-axis aligns with the walking direction (PCA).
    6. Differentiate position to velocity or acceleration (if requested).

    Parameters
    ----------
    data : pd.DataFrame
        Raw motion capture data as loaded from disk.  For ``TRIM_MODE="cone"``,
        must contain walkway cone marker columns (``start_1/2``, ``end_1/2``).
    sub_id : str
        Subject identifier (e.g. ``'pp065'``).  Used for logging and
        per-subject preprocessing parameters.
    task_name : str
        Task name (e.g. ``'walkStroop'``).
    run : str
        Run condition (``'on'`` or ``'off'``).
    tracksys : str
        Tracking system (e.g. ``'omc'``).  Determines which preprocessing
        steps are applied (currently gap-filling and rotation are OMC-only).
    kinematics : str
        Signal type for kinectome computation: ``'pos'``, ``'vel'``, or ``'acc'``.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    pd.DataFrame or None
        Preprocessed data ready for kinectome computation, or ``None`` if
        any step fails (e.g. missing markers, too many NaNs).
    """

    from config import TRIM_MODE

    # ── Step 1: Trim to active walking window ─────────────────────────────────
    # If frame_start/frame_end were already detected (from event detection),
    # pass them directly to avoid running find_walkway_bounds a second time.
    trimmed_data = trim_data.trim_to_walking_window(
        data, sub_id, task_name, run, mode=TRIM_MODE, fs=fs,
        frame_start=frame_start, frame_end=frame_end
    )

    if trimmed_data is None or trimmed_data.empty:
        return None

    # ── Step 2: Remove long NaN streaks ──────────────────────────────────────
    trimmed_data, nan_idx = trim_data.remove_long_nans(
        trimmed_data, sub_id, task_name, run
    )

    if trimmed_data is None or trimmed_data.empty:
        return None

    # ── Step 3: Reduce cluster markers to single points ───────────────────────
    reduced_data = trim_data.reduce_dimensions_clusters(trimmed_data, sub_id, task_name)

    if reduced_data is None or reduced_data.empty:
        return None

    # ── Step 4: Gap-fill and filter (OMC only) ────────────────────────────────
    if tracksys == 'omc':
        interpolated_data = interpolate.fill_gaps(
            reduced_data, sub_id, task_name, fc=6, threshold=271
        )
        # fc = cut-off for the Butterworth filter
        # threshold = maximum allowed gap length in frames

        # ── Step 5: Rotate so x-axis aligns with walking direction (PCA) ──────
        rotated_data = align.rotate_data(interpolated_data, sub_id, task_name)
    else:
        rotated_data = reduced_data

    if rotated_data is None or rotated_data.empty:
        return None

    # ── Step 6: Differentiate to requested kinematic signal ───────────────────
    if kinematics == 'pos':
        diff_data = rotated_data
    elif kinematics == 'vel':
        diff_data = differentiation.velocity(rotated_data, fs)
    elif kinematics == 'acc':
        diff_data = differentiation.acceleration(rotated_data, fs)
    else:
        raise ValueError(
            f"Unknown kinematics '{kinematics}'. "
            "Expected 'pos', 'vel', or 'acc'."
        )

    return diff_data