"""
Gait Event Detection from Optical Motion Capture Data
======================================================
Detects Initial Contacts (IC) from left and right heel markers using the
Bonci et al. (2022) method, ported from MATLAB.

Key features:
  - Walkway gating: ICs are only detected between the start and end cone
    markers (start_1/start_2 and end_1/end_2), automatically derived from
    the marker positions in each file — no hard-coding.
  - Outputs a BIDS-style events TSV with columns:
        onset  duration  event_type
    where event_type is one of:
        start | stop | initial_contact_left | initial_contact_right
  - Outputs an inspection PNG plot (white background).
  - Optionally overlays a reference events TSV for algorithm comparison.

Reference:
    Bonci et al. (2022).
    An Algorithm for Accurate Marker-Based Gait Event Detection in 
    Healthy and Pathological Populations During Complex Motor Tasks
    Front. Bioeng. Biotechnol., 02 June 2022
    https://doi.org/10.3389/fbioe.2022.868928

Author: Ported to Python from MATLAB (Bonci, MOBILISE-D project)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# The constants below are used when running this file directly as a script.
# They are not used when imported as a module by the pipeline.
# PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
# DATA_DIR   = Path(r"C:\Users\Karolina\Desktop\dual\data")
# OUTPUT_DIR = Path(r"C:\Users\Karolina\Desktop\dual\pyevents")
# TSV_FILE   = "sub-pp011_task-walkStroop_tracksys-omc_motion.tsv"
# 
# FS      = 200    # Sampling frequency [Hz]
# FC      = 7      # Low-pass filter cutoff [Hz]
# MM_TO_M = True   # Data is in mm → convert to m
# 
# Reference events TSV for comparison plot (set to None to skip)
# REF_EVENTS_TSV = DATA_DIR / "sub-pp001_task-walkFast_events.tsv"
# REF_EVENTS_TSV = None
# REF_LABEL      = "Romijnders"   # short name shown in legend / title
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Signal processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def butter_lowpass(data: np.ndarray, cutoff: float, fs: float,
                   order: int = 4) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass filter (mirrors MATLAB filtfilt)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    if len(data) > 3 * max(len(a), len(b)):
        return filtfilt(b, a, data, axis=0)
    return data


def filter_marker(traj: np.ndarray, fs: float, fc: float,
                  mm_to_m: bool) -> np.ndarray:
    """
    Convert units (optional) and low-pass filter a [N x 3] marker trajectory.
    Filters each continuous non-NaN segment independently (mirrors dataFiltering.m).
    """
    traj = traj.copy().astype(float)
    if mm_to_m:
        traj /= 1000.0
    nan_mask = np.isnan(traj[:, 0])
    if nan_mask.all():
        return traj

    out      = traj.copy()
    changes  = np.where(np.diff(nan_mask.astype(int)))[0] + 1
    segments = np.split(np.arange(len(traj)), changes)
    for seg in segments:
        if len(seg) < 1 or nan_mask[seg[0]]:
            continue
        if len(seg) > 50:   # mirrors MATLAB's >50 sample guard
            out[seg, :] = butter_lowpass(traj[seg, :], fc, fs)
        else:
            out[seg, :] = traj[seg, :]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Walkway gating — start / end cone detection
# ══════════════════════════════════════════════════════════════════════════════

def find_walkway_bounds(df: pd.DataFrame, mm_to_m: bool, fs: float = 200) -> tuple[int, int]:
    """
    Determine the active frame range for gait event detection.

    Strategy
    --------
    1. Compute the midpoint of the two start markers and the two end markers.
    2. Define the walking direction as the unit vector from start_mid to end_mid.
    3. Project both heel markers onto that direction.
    4. frame_start = first frame where *either* heel projection exceeds 0
       (heel has crossed the start line).
    5. frame_end   = first frame where *either* heel projection exceeds the
       full walkway length (heel past the end line).
       If the end line is never crossed (e.g. back-and-forth / turning tasks),
       frame_end falls back to the last frame of the recording and a warning
       is printed. Detection continues normally in both cases.

    Returns
    -------
    frame_start, frame_end : int
        0-based frame indices (inclusive) into the original signal.
    """
    N = len(df)

    # Check cone marker columns are present before accessing them
    cone_cols = ['start_1_POS_x', 'start_1_POS_y', 'start_2_POS_x', 'start_2_POS_y',
                 'end_1_POS_x',   'end_1_POS_y',   'end_2_POS_x',   'end_2_POS_y']
    missing = [c for c in cone_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cone marker column(s) missing from data: {missing}. "
            "These are required for walkway gate detection (TRIM_MODE='cone'). "
            "If your data has no cone markers, set TRIM_MODE='none' in config.py."
        )

    # midpoints of the two cone pairs (cones are static -- use first frame)
    start_mid = np.array([
        (df['start_1_POS_x'].iloc[0] + df['start_2_POS_x'].iloc[0]) / 2,
        (df['start_1_POS_y'].iloc[0] + df['start_2_POS_y'].iloc[0]) / 2,
    ])
    end_mid = np.array([
        (df['end_1_POS_x'].iloc[0] + df['end_2_POS_x'].iloc[0]) / 2,
        (df['end_1_POS_y'].iloc[0] + df['end_2_POS_y'].iloc[0]) / 2,
    ])

    walk_vec  = end_mid - start_mid
    walk_len  = np.linalg.norm(walk_vec)
    walk_unit = walk_vec / walk_len

    print(f"  Walkway length from markers: {walk_len / 1000:.3f} m")

    # Project both heels onto the walking direction (XY plane only)
    l_heel_xy = df[['l_heel_POS_x', 'l_heel_POS_y']].values.astype(float)
    r_heel_xy = df[['r_heel_POS_x', 'r_heel_POS_y']].values.astype(float)

    l_proj = (l_heel_xy - start_mid) @ walk_unit
    r_proj = (r_heel_xy - start_mid) @ walk_unit

    # Merge: prefer left heel, fall back to right where left is NaN
    proj = np.where(np.isnan(l_proj), r_proj, l_proj)

    # -- Start crossing (mandatory) -------------------------------------------
    past_start = np.where(proj > 0)[0]
    if len(past_start) == 0:
        raise ValueError(
            "No frames found where a heel crosses the start line. "
            "Check that start_1/start_2 marker columns are present and named correctly.")
    frame_start = int(past_start[0])

    # -- End crossing (optional) ----------------------------------------------
    past_end = np.where(proj > walk_len)[0]
    if len(past_end) == 0:
        frame_end = N - 1
        print(f"  NOTE: end line never crossed (back-and-forth / turning task). "
              f"Gate extends to last frame ({frame_end}).")
    else:
        frame_end = int(past_end[0])

    print(f"  Walkway gate: frames {frame_start} to {frame_end}  "
          f"({(frame_end - frame_start) / fs:.2f} s)")

    return frame_start, frame_end


# ══════════════════════════════════════════════════════════════════════════════
# Pelvis reference
# ══════════════════════════════════════════════════════════════════════════════

def compute_pelvis_ap(df: pd.DataFrame, mm_to_m: bool) -> np.ndarray | None:
    """Mid-pelvis AP reference for IC detection.

    Uses mid-PSIS x as the primary reference (mirrors GE_Zeni.m).
    If one or both PSIS markers are missing or fully NaN, falls back to
    mid-ASIS x, then to the average of all available pelvic markers.
    Any remaining NaN gaps are filled by linear interpolation.

    All four pelvic markers (L/R ASIS, L/R PSIS) are rigidly connected,
    so any available subset gives a valid AP reference.

    Parameters
    ----------
    df : pd.DataFrame
        Raw motion capture data.
    mm_to_m : bool
        Convert from mm to m before returning.

    Returns
    -------
    np.ndarray or None
        Pelvis AP signal, or None if no pelvic markers are found at all.
    """
    pelvic_candidates = [
        ['l_psis_POS_x', 'r_psis_POS_x'],   # preferred: mid-PSIS
        ['l_asis_POS_x', 'r_asis_POS_x'],   # fallback: mid-ASIS
        ['l_psis_POS_x'],                    # single PSIS
        ['r_psis_POS_x'],
        ['l_asis_POS_x'],                    # single ASIS
        ['r_asis_POS_x'],
    ]

    mid_x = None
    used = None
    for cols in pelvic_candidates:
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        signals = np.array([df[c].values.astype(float) for c in available])
        candidate = np.nanmean(signals, axis=0)
        if not np.all(np.isnan(candidate)):
            mid_x = candidate
            used = available
            break

    if mid_x is None:
        return None

    if len(used) < 2:
        print(f"  WARNING: only {used[0]} available for pelvis AP reference.")
    elif used != ['l_psis_POS_x', 'r_psis_POS_x']:
        print(f"  NOTE: using {used} as pelvis AP reference (preferred PSIS markers unavailable).")

    if mm_to_m:
        mid_x = mid_x / 1000.0

    return pd.Series(mid_x).interpolate(method='linear', limit_direction='both').values


# ══════════════════════════════════════════════════════════════════════════════
# IC detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_IC_bonci(heel_x: np.ndarray, pelvis_x: np.ndarray,
                   fs: float) -> np.ndarray:
    """
    Detect Initial Contacts using Bonci et al. (2022).

    ICs are positive peaks of (heel_x - pelvis_x) where the velocity
    transitions from positive to negative (zero-crossing at the peak).

    Parameters
    ----------
    heel_x   : filtered AP heel position [m], full signal length
    pelvis_x : AP pelvis reference [m],       full signal length
    fs       : sampling frequency [Hz]

    Returns
    -------
    IC : 0-based frame indices (in the full signal) of detected ICs
    """
    rel      = heel_x - pelvis_x
    peaks, _ = find_peaks(rel, prominence=0.05)
    vel      = np.diff(rel) * fs

    IC = []
    for p in peaks:
        if p == 0 or p >= len(vel):
            continue
        if vel[p - 1] > 0 and vel[p] < 0:
            IC.append(p)
    return np.array(IC, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# Main processing pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_and_process(tsv_path: Path, fs: float, fc: float,
                     mm_to_m: bool) -> tuple[dict, int, int, int]:
    """
    Load TSV, filter markers, find walkway gate, detect ICs within the gate.

    Returns
    -------
    results     : dict with per-side data and IC indices (full-signal frame numbers)
    N           : total number of frames in the file
    frame_start : first frame inside the walkway gate
    frame_end   : last frame inside the walkway gate
    """
    print(f"Loading: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    N  = len(df)
    print(f"  Total frames: {N}  ({N / fs:.1f} s at {fs} Hz)")

    # ── Walkway gate ─────────────────────────────────────────────────────────
    frame_start, frame_end = find_walkway_bounds(df, mm_to_m, fs=fs)

    # ── Pelvis reference (full signal, then gated slice used for IC) ─────────
    pelvis_x_full = compute_pelvis_ap(df, mm_to_m)

    results = {}
    for side, prefix in [('left', 'l'), ('right', 'r')]:
        heel_cols = [f'{prefix}_heel_POS_x',
                     f'{prefix}_heel_POS_y',
                     f'{prefix}_heel_POS_z']

        if not all(c in df.columns for c in heel_cols):
            print(f"  WARNING: heel columns missing for {side} — skipping.")
            continue

        heel_raw = df[heel_cols].values.astype(float)
        heel_flt = filter_marker(heel_raw, fs, fc, mm_to_m)   # full signal

        # Pelvis fallback
        if pelvis_x_full is None:
            print("  WARNING: pelvis markers not found — using mean heel x.")
            pelvis_x = np.full(N, np.nanmean(heel_flt[:, 0]))
        else:
            pelvis_x = pelvis_x_full

        # Detect ICs on the full signal, then keep only those inside the gate
        IC_all = detect_IC_bonci(heel_flt[:, 0], pelvis_x, fs)
        IC = IC_all[(IC_all >= frame_start) & (IC_all <= frame_end)]

        print(f"  {side.capitalize()} ICs in walkway gate: {len(IC)}  "
              f"(of {len(IC_all)} total)")

        results[side] = {
            'heel_x_flt': heel_flt[:, 0],
            'pelvis_x'  : pelvis_x,
            'rel_x'     : heel_flt[:, 0] - pelvis_x,
            'IC'        : IC,
        }

    return results, N, frame_start, frame_end


# ══════════════════════════════════════════════════════════════════════════════
# TSV output
# ══════════════════════════════════════════════════════════════════════════════

def save_events_tsv(results: dict, frame_start: int, frame_end: int,
                    output_path: Path) -> None:
    """
    Save detected events as a BIDS-style events TSV.

    Columns : onset  duration  event_type
    onset   : frame index (0-based, same reference as the motion TSV)
    duration: 0  (instantaneous events)
    event_type: start | stop | initial_contact_left | initial_contact_right

    Events are sorted by onset.
    """
    rows = [
        {'onset': frame_start, 'duration': 0, 'event_type': 'start'},
        {'onset': frame_end,   'duration': 0, 'event_type': 'stop'},
    ]
    for side in ['left', 'right']:
        if side not in results:
            continue
        label = f'initial_contact_{side}'
        for idx in results[side]['IC']:
            rows.append({'onset': int(idx), 'duration': 0, 'event_type': label})

    events_df = (pd.DataFrame(rows)
                 .sort_values('onset')
                 .reset_index(drop=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(output_path, sep='\t', index=False)
    print(f"Events TSV saved: {output_path}  ({len(events_df)} events)")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers — shared style
# ══════════════════════════════════════════════════════════════════════════════

STYLE_WHITE = {
    'font.family'       : 'DejaVu Sans',
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
    'axes.edgecolor'    : '#444444',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.labelcolor'   : '#222222',
    'xtick.color'       : '#444444',
    'ytick.color'       : '#444444',
    'text.color'        : '#222222',
    'grid.color'        : '#dddddd',
    'grid.linewidth'    : 0.7,
}

# Trace colours (white-bg versions)
TRACE_COL = {'left': '#1565C0',  'right': '#B71C1C'}    # deep blue / deep red
IC_COL    = {'left': '#E65100',  'right': '#1B5E20'}     # burnt orange / dark green


# ══════════════════════════════════════════════════════════════════════════════
# Standard plot (Bonci ICs only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_gait_events(results: dict, N: int, fs: float,
                     frame_start: int, frame_end: int,
                     output_path: Path,
                     subject: str = '', task: str = '') -> None:
    """
    Inspection plot: heel AP position relative to pelvis with Bonci ICs marked.
    A shaded band shows the gated walkway region.
    White background.
    """
    time = np.arange(N) / fs
    plt.rcParams.update(STYLE_WHITE)

    sides   = [s for s in ['left', 'right'] if s in results]
    n_sides = len(sides)

    fig = plt.figure(figsize=(16, 4.5 * n_sides), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n_sides, 1, figure=fig)

    for row, side in enumerate(sides):
        ax  = fig.add_subplot(gs[row])
        res = results[side]
        rel = res['rel_x']
        IC  = res['IC']

        # walkway gate shading
        ax.axvspan(frame_start / fs, frame_end / fs,
                   color='#E3F2FD', alpha=0.6, label='5m walkway', zorder=0)

        # trajectory
        ax.plot(time, rel, color=TRACE_COL[side], lw=1.2, alpha=0.85,
                label='Heel – Pelvis (AP)')

        # IC markers
        if len(IC) > 0:
            ax.scatter(time[IC], rel[IC],
                       color=IC_COL[side], s=160, zorder=5, marker='v',
                       edgecolors='#222222', linewidths=0.4,
                       label=f'Initial Contact  (n={len(IC)})')
            for idx in IC:
                ax.axvline(time[idx], color=IC_COL[side],
                           lw=0.6, alpha=0.45, linestyle='--')

        # start / stop lines
        ax.axvline(frame_start / fs, color='#2E7D32', lw=1.4,
                   linestyle='-', label='Start')
        ax.axvline(frame_end   / fs, color='#C62828', lw=1.4,
                   linestyle='-', label='Stop')

        ax.set_title(f'{side.capitalize()} Heel — AP Position relative to Pelvis',
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel('Relative AP position [m]', fontsize=10)
        ax.legend(framealpha=0.85, fontsize=9, loc='upper right',
                  edgecolor='#cccccc')
        ax.grid(True, axis='both')
        ax.text(0.01, 0.97, f'ICs in gate: {len(IC)}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color=IC_COL[side])

    title = 'Gait Event Detection — Initial Contacts (Bonci 2022)'
    if subject or task:
        title += f'\n{subject}  |  {task}  |  fs = {fs} Hz'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved: {output_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot (Bonci ICs + Romijnders ICs)
# ══════════════════════════════════════════════════════════════════════════════

def load_reference_events(events_tsv: Path) -> dict:
    """
    Load a BIDS-style events TSV and return IC frame indices per side.
    Returns dict with keys 'left' and 'right'.
    """
    df  = pd.read_csv(events_tsv, sep='\t')
    ref = {}
    for side in ['left', 'right']:
        label      = f'initial_contact_{side}'
        ref[side]  = df.loc[df['event_type'] == label, 'onset'].values.astype(int)
    return ref


def plot_gait_events_comparison(results: dict, N: int, fs: float,
                                 frame_start: int, frame_end: int,
                                 ref_events: dict, ref_label: str,
                                 output_path: Path,
                                 subject: str = '', task: str = '') -> None:
    """
    Same as plot_gait_events() with reference ICs overlaid in a distinct
    colour and marker shape for direct algorithm comparison.
    White background.
    """
    time = np.arange(N) / fs
    plt.rcParams.update(STYLE_WHITE)

    sides   = [s for s in ['left', 'right'] if s in results]
    n_sides = len(sides)

    # Reference algorithm colours
    REF_COL = {'left': '#6A1B9A', 'right': '#00838F'}   # purple / teal

    fig = plt.figure(figsize=(16, 4.5 * n_sides), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n_sides, 1, figure=fig)

    for row, side in enumerate(sides):
        ax  = fig.add_subplot(gs[row])
        res = results[side]
        rel = res['rel_x']
        IC  = res['IC']

        # walkway gate shading
        ax.axvspan(frame_start / fs, frame_end / fs,
                   color='#E3F2FD', alpha=0.5, label='5m walkway', zorder=0)

        # trajectory
        ax.plot(time, rel, color=TRACE_COL[side], lw=1.2, alpha=0.85,
                label='Heel – Pelvis (AP)')

        # Bonci ICs
        if len(IC) > 0:
            ax.scatter(time[IC], rel[IC],
                       color=IC_COL[side], s=160, zorder=5, marker='v',
                       edgecolors='#222222', linewidths=0.4,
                       label=f'Bonci IC  (n={len(IC)})')
            for idx in IC:
                ax.axvline(time[idx], color=IC_COL[side],
                           lw=0.6, alpha=0.4, linestyle='--')

        # Reference ICs
        ref_IC = ref_events.get(side, np.array([], dtype=int))
        ref_IC = ref_IC[ref_IC < N]
        if len(ref_IC) > 0:
            ax.scatter(time[ref_IC], rel[ref_IC],
                       color=REF_COL[side], s=180, zorder=6, marker='^',
                       edgecolors='#222222', linewidths=0.5,
                       label=f'{ref_label} IC  (n={len(ref_IC)})')
            for idx in ref_IC:
                ax.axvline(time[idx], color=REF_COL[side],
                           lw=0.8, alpha=0.4, linestyle=':')

        # start / stop lines
        ax.axvline(frame_start / fs, color='#2E7D32', lw=1.4,
                   linestyle='-', label='Start')
        ax.axvline(frame_end   / fs, color='#C62828', lw=1.4,
                   linestyle='-', label='Stop')

        ax.set_title(f'{side.capitalize()} Heel — AP Position relative to Pelvis',
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel('Relative AP position [m]', fontsize=10)
        ax.legend(framealpha=0.85, fontsize=9, loc='upper right',
                  edgecolor='#cccccc')
        ax.grid(True, axis='both')
        ax.text(0.01, 0.97,
                f'Bonci: {len(IC)}   |   {ref_label}: {len(ref_IC)}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='#333333')

    title = f'IC Comparison: Bonci 2022  vs  {ref_label}'
    if subject or task:
        title += f'\n{subject}  |  {task}  |  fs = {fs} Hz'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Comparison plot saved: {output_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, fs: float,
                  frame_start: int, frame_end: int) -> None:
    print("\n" + "=" * 55)
    print("  GAIT EVENT SUMMARY — Initial Contacts")
    print("=" * 55)
    print(f"  Gate: frame {frame_start} → {frame_end}  "
          f"({(frame_end - frame_start) / fs:.2f} s)")
    for side in ['left', 'right']:
        if side not in results:
            continue
        IC = results[side]['IC']
        print(f"\n  {side.upper()} ({len(IC)} events):")
        if len(IC) == 0:
            print("    — none detected —")
        else:
            print(f"    {'Frame':>7}   {'Time (s)':>9}")
            print("    " + "-" * 22)
            for idx in IC:
                print(f"    {idx:>7}   {idx / fs:>9.3f}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


# The block below runs only when this file is executed directly as a script,
# not when it is imported by the pipeline.
# if __name__ == "__main__":
#     tsv_path = DATA_DIR / TSV_FILE
# 
#     # Derive subject / task labels from filename for titles and output names
#     stem    = Path(TSV_FILE).stem
#     parts   = stem.split('_')
#     subject = next((p for p in parts if p.startswith('sub-')),  '')
#     task    = next((p for p in parts if p.startswith('task-')), '')
#     prefix  = f"{subject}_{task}" if (subject or task) else stem
# 
#     # ── Process ───────────────────────────────────────────────────────────────
#     results, N, frame_start, frame_end = load_and_process(
#         tsv_path, FS, FC, MM_TO_M)
# 
#     print_summary(results, FS, frame_start, frame_end)
# 
#     # ── Save events TSV ───────────────────────────────────────────────────────
#     events_path = OUTPUT_DIR / f"{prefix}_events.tsv"
#     save_events_tsv(results, frame_start, frame_end, events_path)
# 
#     # ── Standard plot ─────────────────────────────────────────────────────────
#     plot_path = OUTPUT_DIR / f"{prefix}_gait_events_IC.png"
#     plot_gait_events(results, N, FS, frame_start, frame_end,
#                      plot_path, subject=subject, task=task)
# 
#     # ── Comparison plot (optional) ────────────────────────────────────────────
#     if REF_EVENTS_TSV is not None and Path(REF_EVENTS_TSV).exists():
#         ref_events   = load_reference_events(REF_EVENTS_TSV)
#         compare_path = OUTPUT_DIR / f"{prefix}_gait_events_IC_comparison.png"
#         plot_gait_events_comparison(results, N, FS, frame_start, frame_end,
#                                     ref_events, REF_LABEL, compare_path,
#                                     subject=subject, task=task)
#     elif REF_EVENTS_TSV is not None:
#         print(f"\nNOTE: reference events file not found at {REF_EVENTS_TSV} — "
#               "skipping comparison plot.")
# 
#     print("\nDone.")

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration — detect directly from a DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def detect_events_from_dataframe(df: pd.DataFrame, fs: float,
                                  fc: float = 7.0,
                                  mm_to_m: bool = True) -> tuple[dict, int, int]:
    """Detect gait events directly from a motion capture DataFrame.

    This is the primary integration point for the pipeline.  It runs the full
    Bonci et al. (2022) detection on an already-loaded DataFrame, avoiding the
    need to write a temporary file to disk.

    Detects:
    - Walkway gate (``frame_start``, ``frame_end``) from cone markers.
    - Left and right initial contacts (ICs) within the gate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw motion capture data.  Must contain:

        - ``l_heel_POS_x/y/z``, ``r_heel_POS_x/y/z`` — heel markers
        - ``l_psis_POS_x``, ``r_psis_POS_x`` — pelvis reference markers
        - ``start_1_POS_x/y``, ``start_2_POS_x/y`` — start cone markers
        - ``end_1_POS_x/y``,   ``end_2_POS_x/y``   — end cone markers

    fs : float
        Sampling frequency in Hz.
    fc : float, optional
        Low-pass filter cutoff in Hz.  Default 7 Hz (Bonci et al. 2022).
    mm_to_m : bool, optional
        Convert marker positions from mm to m before processing.  Default True.

    Returns
    -------
    results : dict
        Per-side detection results.  Keys: ``'left'``, ``'right'``.
        Each value is a dict with key ``'IC'``: array of 0-based frame indices
        of detected initial contacts (in full-signal frame space).
    frame_start : int
        First frame of the walkway gate (0-based).
    frame_end : int
        Last frame of the walkway gate (0-based).
    """
    N = len(df)

    # Walkway gate from cone markers
    frame_start, frame_end = find_walkway_bounds(df, mm_to_m, fs=fs)

    # Pelvis AP reference (full signal)
    pelvis_x_full = compute_pelvis_ap(df, mm_to_m)

    results = {}
    for side, prefix in [('left', 'l'), ('right', 'r')]:
        heel_cols = [f'{prefix}_heel_POS_x',
                     f'{prefix}_heel_POS_y',
                     f'{prefix}_heel_POS_z']

        if not all(c in df.columns for c in heel_cols):
            print(f"  WARNING: heel columns missing for {side} — skipping.")
            continue

        heel_raw = df[heel_cols].values.astype(float)
        heel_flt = filter_marker(heel_raw, fs, fc, mm_to_m)

        if pelvis_x_full is None:
            print("  WARNING: pelvis markers not found — using mean heel x.")
            pelvis_x = np.full(N, np.nanmean(heel_flt[:, 0]))
        else:
            pelvis_x = pelvis_x_full

        IC_all = detect_IC_bonci(heel_flt[:, 0], pelvis_x, fs)
        IC = IC_all[(IC_all >= frame_start) & (IC_all <= frame_end)]

        print(f"  {side.capitalize()} ICs in gate: {len(IC)}")
        results[side] = {'IC': IC}

    return results, frame_start, frame_end