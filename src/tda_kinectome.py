"""
tda_kinectome.py — Topological Data Analysis (TDA) of kinectome graphs.
=======================================================================

Persistent homology of kinectome connectivity matrices using giotto-tda
(https://giotto-ai.github.io/gtda-docs/). Requires ``pip install giotto-tda``.

Each kinectome is a complete, undirected, weighted graph whose entries are
correlations in [-1, 1] (1.0 on the diagonal). For full kinectomes the graph
has n_markers*3 vertices (e.g. 48 for walkStroop, 66 otherwise); for
directional kinectomes each of AP/ML/V is analysed as its own graph.

Filter function & filtration
----------------------------
Edges are filtered by absolute weight: for threshold ``t`` only edges with
``|w| >= t`` are kept. This is translated into a genuine sub-level-set
filtration via the distance

    d(i, j) = 1 - |w(i, j)|            in [0, 1]

so keeping ``|w| >= t`` is exactly keeping ``d <= 1 - t``. Building the flag
(clique) complex on the sub-level sets of ``d`` yields the full persistence
diagram in one shot (FlagserPersistence), and Betti numbers are read off at
the requested thresholds ``t``.

This module follows the pipeline's conventions
-----------------------------------------------
- All parameters come from ``config.py`` (TDA_* and the shared paths).
- Data is discovered through ``load_kinectomes`` using the same age-matched
  groups (``get_matched_groups_for_task``) and the same task-specific marker
  exclusion (``EXCLUDE_MARKERS_BY_TASK``) as every other module — it never
  walks the disk blindly.
- Works for full and directional kinectomes (shape-agnostic).
- Two analysis levels (config ``TDA_LEVELS``):
    'cycle'   — every individual per-gait-cycle kinectome (inter-cycle
                topology within a subject).
    'subject' — one averaged kinectome per subject (intra-subject topology,
                comparable across the group).

Outputs (under RESULT_BASE_PATH/tda/<level>/<group>/<direction>/)
-----------------------------------------------------------------
- ``<stem>_betti_curve.png``         : Betti curves b_k vs threshold t.
- ``<stem>_persistence_diagram.png`` : persistence diagrams (d = 1 - |w|).
- ``<stem>_barcode.png``             : persistence barcode along the
  correlation filtration (bottom axis d = 1 - |w|, top axis |w|). Long bars =
  features robust across a wide correlation range; short bars = noise.

Cross-cycle survival (needs cycle-level data), under RESULT_BASE_PATH/tda/cross_cycle/
-------------------------------------------------------------------------------------
Within-subject stride-to-stride topology, one figure per subject/stratum:
- top panel:    Betti number per gait cycle at a fixed threshold (does the
                amount of cyclic structure hold across strides?).
- bottom panel: bottleneck distance between consecutive cycles' diagrams — a
                topological stride-to-stride variability trace. Its mean is a
                per-subject scalar rolled up into the group comparison.

Group comparison (subject level only), under RESULT_BASE_PATH/tda/group_comparison/
-----------------------------------------------------------------------------------
Each averaged (subject-level) diagram is reduced to scalar features per
homology dimension — number of features, total / max / mean persistence, and
persistence entropy — and the two age-matched groups are compared, separately
per (task, kinematic, direction) stratum:
    - Shapiro-Wilk normality per group; both normal -> Welch's t-test with
      Cohen's d, otherwise Mann-Whitney U with rank-biserial correlation.
    - Benjamini-Hochberg FDR across features within each stratum.
Additionally per stratum:
    - persistence images vectorise each subject's diagram; a pixel-wise
      PD-vs-control test (FDR across pixels) gives a signed difference heatmap
      showing *where* in birth-persistence space the groups diverge, and a
      cross-validated logistic-regression classifier on the image vectors
      gives an AUC with a label-permutation p-value (*how separable* they are).
    - persistence landscapes vectorise each diagram into a vector space where
      the group *mean* is well-defined (the averageable stand-in for an
      'average barcode'): a group-mean landscape figure per dimension (PD vs
      control, layer 1 with +/- SEM band) plus a permutation test on the L2
      distance between the two group-mean landscapes, added to the table as
      'H<k>_landscape_L2'.
    - if cycle-level data exist, the per-subject mean consecutive-cycle
      bottleneck distance (topological stride-to-stride variability) is added
      to the test table as 'cross_cycle_bottleneck_mean'.
Outputs:
- ``group_comparison/tda_group_comparison.csv`` / ``.pkl`` : full stats table
  (scalar features + persistence_image_classifier_AUC + H<k>_landscape_L2 +
  cross_cycle_bottleneck_mean).
- ``group_comparison/betti_group_<task>_<kin>_<dir>_H<k>.png`` : group-mean
  Betti curves (mean +/- SEM) with FDR-significant thresholds starred.
- ``group_comparison/pi_diff_<task>_<kin>_<dir>.png`` : persistence-image
  difference heatmap (disease - control) with FDR-significant pixels outlined.
- ``group_comparison/landscape_group_<task>_<kin>_<dir>_H<k>.png`` : group-mean
  persistence landscapes (PD vs control) per homology dimension.
Cycle-level graphs are never pooled across subjects (not independent), so the
subject-level tests use averaged kinectomes; cycle data feed only the
within-subject cross-cycle variability scalar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_utils.data_loader import load_kinectomes, exclude_markers_from_kinectome
from src.data_utils.groups import get_matched_groups_for_task


_DIRS3 = ["AP", "ML", "V"]
DIM_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green",
              3: "tab:red", 4: "tab:purple"}


# --------------------------------------------------------------------------- #
# Matrix preparation
# --------------------------------------------------------------------------- #

def weights_to_distance(weights: np.ndarray) -> np.ndarray:
    """Convert a signed 2D weight matrix into a [0, 1] distance d = 1 - |w|.
    Symmetry is enforced and the diagonal set to 0 (all vertices present from
    the start)."""
    w = np.asarray(weights, dtype=float)
    w = (w + w.T) / 2.0
    dist = 1.0 - np.abs(w)
    np.clip(dist, 0.0, 1.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def _matrix_directions(kinectome, full):
    """Yield (direction_label, 2D_matrix) for one stored kinectome.
    Full -> [('full', M)]; directional (3D) -> [('AP',..),('ML',..),('V',..)].
    A 2D array is always treated as a single 'full' graph regardless of the
    flag, since it has no separable directions."""
    arr = np.asarray(kinectome, dtype=float)
    if arr.ndim == 2:
        yield ('full', arr)
    else:
        for i in range(arr.shape[2]):
            yield (_DIRS3[i], arr[:, :, i])


# --------------------------------------------------------------------------- #
# Persistent homology / Betti numbers
# --------------------------------------------------------------------------- #

def compute_persistence_diagrams(distance_matrices, homology_dimensions,
                                 infinity_sentinel):
    """Persistence diagrams for a batch of 2D distance matrices via
    FlagserPersistence (undirected flag complex, dims as configured)."""
    from gtda.homology import FlagserPersistence  # imported lazily
    model = FlagserPersistence(
        homology_dimensions=homology_dimensions,
        directed=False,
        filtration="max",
        max_edge_weight=1.0,
        infinity_values=infinity_sentinel,
        reduced_homology=False,
        n_jobs=-1,
    )
    stacked = np.stack(distance_matrices, axis=0)
    return list(model.fit_transform(stacked))


def betti_curve_from_diagram(diagram, homology_dimensions, distance_values):
    """b_k(d) = #{(birth, death) of dim k : birth <= d < death}, at each
    d = 1 - t. Returns {k: counts_over_thresholds}."""
    births, deaths, dims = diagram[:, 0], diagram[:, 1], diagram[:, 2]
    betti = {}
    for k in homology_dimensions:
        mask_k = dims == k
        b_k, d_k = births[mask_k], deaths[mask_k]
        betti[k] = np.array(
            [np.sum((b_k <= d) & (d_k > d)) for d in distance_values], dtype=int
        )
    return betti


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_betti_curves(betti_by_dim, filter_thresholds, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, counts in betti_by_dim.items():
        ax.step(filter_thresholds, counts, where="post", marker="o",
                label=f"$b_{{{k}}}$", color=DIM_COLORS.get(k))
    ax.set_xlabel(r"Filter threshold $t$  (edges kept when $|w_{ij}| \geq t$)")
    ax.set_ylabel("Betti number")
    ax.set_title(f"Betti curves\n{title}")
    ax.set_xticks(filter_thresholds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_persistence_diagrams(diagram, homology_dimensions, infinity_sentinel,
                              out_path, title):
    n_dim = len(homology_dimensions)
    ncols = 3
    nrows = int(np.ceil(n_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_dim:]:
        ax.axis("off")
    for ax, k in zip(axes, homology_dimensions):
        mask = diagram[:, 2] == k
        b, d = diagram[mask, 0], diagram[mask, 1]
        essential = d >= (infinity_sentinel - 1e-9)
        finite = ~essential
        ax.scatter(b[finite], d[finite], color=DIM_COLORS.get(k), alpha=0.7,
                   label=f"H{k}")
        if essential.any():
            ax.scatter(b[essential], np.full(essential.sum(), 1.0), marker="^",
                       color=DIM_COLORS.get(k), edgecolor="black", s=80,
                       label=f"H{k} (never dies)")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Birth ($d = 1 - |w|$)")
        ax.set_ylabel("Death ($d = 1 - |w|$)")
        ax.set_title(f"H{k}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Persistence diagrams\n{title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_barcode(diagram, homology_dimensions, infinity_sentinel, out_path,
                 title):
    """Persistence barcode with the correlation filtration as the x-axis.

    Each homology class is one horizontal bar from birth to death in
    ``d = 1 - |w|`` (bottom axis). A secondary top axis relabels the same
    positions in correlation strength ``|w| = 1 - d`` so the bar can be read
    as "the range of correlation strengths over which this feature exists".
    Longer bars = features that persist across a wide correlation range
    (robust structure); short bars = likely noise. Essential classes (death =
    sentinel) are drawn to d = 1.0 with an open arrow head.

    Bars are grouped and coloured by dimension, sorted by birth within each
    dimension so the longest-lived features are easy to spot.
    """
    births, deaths, dims = diagram[:, 0], diagram[:, 1], diagram[:, 2]

    # Collect bars per dimension, dropping zero-persistence padding rows.
    ordered = []  # (dim, birth, death, essential)
    for k in homology_dimensions:
        mask = dims == k
        b_k, d_k = births[mask], deaths[mask]
        keep = d_k > b_k
        b_k, d_k = b_k[keep], d_k[keep]
        order = np.argsort(b_k)
        for b, d in zip(b_k[order], d_k[order]):
            essential = d >= (infinity_sentinel - 1e-9)
            ordered.append((k, float(b), 1.0 if essential else float(d),
                            essential))

    if not ordered:
        fig, ax = plt.subplots(figsize=(7.5, 3))
        ax.text(0.5, 0.5, "no persistent features", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    height = max(3.0, 0.16 * len(ordered) + 1.2)
    fig, ax = plt.subplots(figsize=(8, height))
    seen_dims = set()
    for y, (k, b, d, essential) in enumerate(ordered):
        colour = DIM_COLORS.get(k)
        label = f"H{k}" if k not in seen_dims else None
        seen_dims.add(k)
        ax.plot([b, d], [y, y], color=colour, linewidth=2.4, solid_capstyle="butt",
                label=label)
        if essential:
            ax.plot(d, y, marker=">", color=colour, markersize=7)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-1, len(ordered))
    ax.set_yticks([])
    ax.set_xlabel(r"Filtration $d = 1 - |w|$  (feature born at strong $|w|$, dies as $|w|$ weakens)")
    ax.grid(True, axis="x", alpha=0.3)

    # Top axis in correlation strength |w| = 1 - d (runs 1 -> 0).
    secax = ax.secondary_xaxis("top", functions=(lambda x: 1 - x, lambda x: 1 - x))
    secax.set_xlabel(r"Correlation strength $|w| = 1 - d$")

    # De-duplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=9)
    ax.set_title(f"Persistence barcode\n{title}", pad=28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data assembly (matrices + labels) following pipeline conventions
# --------------------------------------------------------------------------- #

def _gather_matrices(diagnosis, kinematics_list, task_names, tracking_systems,
                     runs, pd_on, full, correlation_method, marker_list_affect,
                     levels):
    """Collect the 2D matrices to analyse, tagged with (level, group, sub_id,
    task, kinematic, direction, index). Returns a list of dicts:
        {'matrix': 2D ndarray, 'stem': str, 'subdir': Path-relative str}
    """
    from config import KINECTOME_SAVE_PATH, EXCLUDE_MARKERS_BY_TASK

    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    all_disease = sorted(set(s for ids in task_disease_ids.values() for s in ids))
    all_control = sorted(set(s for ids in task_control_ids.values() for s in ids))
    pk_name = f"{diagnosis[0][10:].capitalize()}"
    ctrl_name = "Control"

    items = []

    for kinematics in kinematics_list:
        for sub_id in all_disease + all_control:
            group = pk_name if sub_id in all_disease else ctrl_name
            for tracksys in tracking_systems:
                for task_name in task_names:
                    # per-task matched membership
                    if sub_id in all_disease and sub_id not in task_disease_ids.get(task_name, []):
                        continue
                    if sub_id in all_control and sub_id not in task_control_ids.get(task_name, []):
                        continue
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in all_disease:
                            run = None

                        kinectomes = load_kinectomes(
                            KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run,
                            kinematics, full, correlation_method
                        )
                        if not kinectomes:
                            continue

                        # marker exclusion (full-aware)
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        if exclude:
                            if full:
                                labels = [f"{m}_{d}" for m in marker_list_affect for d in _DIRS3]
                                excl = [f"{m}_{d}" for m in exclude for d in _DIRS3]
                            else:
                                labels = list(marker_list_affect)
                                excl = list(exclude)
                            reduced = []
                            for k in kinectomes:
                                kr, labels = exclude_markers_from_kinectome(k, labels, excl)
                                reduced.append(kr)
                            kinectomes = reduced

                        # ---- level: subject (mean over cycles) ----
                        if 'subject' in levels:
                            mean_k = np.nanmean(np.stack(kinectomes, axis=0), axis=0)
                            for dlabel, dmat in _matrix_directions(mean_k, full):
                                items.append({
                                    'matrix': weights_to_distance(dmat),
                                    'level': 'subject', 'group': group,
                                    'direction': dlabel,
                                    'stem': f"{sub_id}_{task_name}_{kinematics}_{dlabel}_mean",
                                    'subdir': Path('subject') / group / dlabel,
                                })

                        # ---- level: cycle (each cycle separately) ----
                        if 'cycle' in levels:
                            for ci, cyc in enumerate(kinectomes):
                                for dlabel, dmat in _matrix_directions(cyc, full):
                                    items.append({
                                        'matrix': weights_to_distance(dmat),
                                        'level': 'cycle', 'group': group,
                                        'direction': dlabel,
                                        'stem': f"{sub_id}_{task_name}_{kinematics}_{dlabel}_cycle{ci:03d}",
                                        'subdir': Path('cycle') / group / dlabel / f"sub-{sub_id}",
                                    })

    return items


# --------------------------------------------------------------------------- #
# Diagram <-> giotto batch conversion (padding to equal length)
# --------------------------------------------------------------------------- #

def _stack_diagrams(diagram_list, homology_dimensions):
    """Pad a list of (n_i, 3) diagrams and stack into the (n_samples, n_points,
    3) array giotto-tda expects.

    giotto requires the *same number of triples per homology dimension* across
    all diagrams (not just equal total length), so padding is done per
    dimension with trivial (b, b, dim) points (zero persistence) that every
    giotto transformer ignores. The padding birth is set to each diagram's own
    max birth in that dimension (0.0 if none), keeping the trivial points on
    the diagonal."""
    if not diagram_list:
        return None
    dims = list(homology_dimensions)

    # Max count of points per dimension across the whole batch.
    max_per_dim = {}
    for k in dims:
        max_per_dim[k] = max(int(np.sum(d[:, 2] == k)) for d in diagram_list)

    padded = []
    for d in diagram_list:
        blocks = []
        for k in dims:
            sub = d[d[:, 2] == k]
            need = max_per_dim[k] - sub.shape[0]
            if need > 0:
                # trivial diagonal points (birth = death) for this dimension
                fill_val = float(sub[:, 0].max()) if sub.shape[0] else 0.0
                pad = np.column_stack([
                    np.full(need, fill_val), np.full(need, fill_val),
                    np.full(need, float(k))])
                sub = np.vstack([sub, pad]) if sub.shape[0] else pad
            blocks.append(sub)
        padded.append(np.vstack(blocks))
    return np.stack(padded, axis=0)


# --------------------------------------------------------------------------- #
# Persistence images: vectorise diagrams, pixel-wise test + classifier
# --------------------------------------------------------------------------- #

def persistence_images(diagram_list, homology_dimensions, n_bins=20):
    """Vectorise a list of diagrams into persistence images.
    Returns (images, flat) where images is
    (n_samples, n_dims, n_bins, n_bins) and flat is (n_samples, n_features)."""
    from gtda.diagrams import PersistenceImage
    X = _stack_diagrams(diagram_list, homology_dimensions)
    pi = PersistenceImage(n_bins=n_bins)
    images = pi.fit_transform(X)
    flat = images.reshape(images.shape[0], -1)
    return images, flat


def pi_pixelwise_difference(img_disease, img_control):
    """Pixel-wise PD-vs-control comparison of persistence images.
    ``img_*`` are (n_subjects, n_dims, n_bins, n_bins). For each pixel runs the
    distribution-aware two-sample test, FDR-corrects across all pixels (all
    dims together), and returns per-dimension arrays:
        {k: {'effect': (n_bins,n_bins), 'p': ..., 'q': ..., 'sig': ...}}
    Effect is disease - control mean (signed), useful as a difference heatmap.
    """
    n_dims = img_disease.shape[1]
    n_bins = img_disease.shape[2]
    # Collect p-values across every pixel and dim, then one global FDR.
    p_all, coords = [], []
    diff_by_dim = {}
    for di in range(n_dims):
        diff_by_dim[di] = np.full((n_bins, n_bins), np.nan)
        for r in range(n_bins):
            for c in range(n_bins):
                d = img_disease[:, di, r, c]
                ct = img_control[:, di, r, c]
                diff_by_dim[di][r, c] = np.nanmean(d) - np.nanmean(ct)
                # Skip pixels that are all-zero in both groups (empty region).
                if np.allclose(d, 0) and np.allclose(ct, 0):
                    continue
                res = compare_feature(d, ct)
                p_all.append(res["p"])
                coords.append((di, r, c))
    rej, q = _fdr_bh(p_all)
    out = {di: {"effect": diff_by_dim[di],
                "p": np.full((n_bins, n_bins), np.nan),
                "q": np.full((n_bins, n_bins), np.nan),
                "sig": np.zeros((n_bins, n_bins), bool)}
           for di in range(n_dims)}
    for (di, r, c), pv, qv, rj in zip(coords, p_all, q, rej):
        out[di]["p"][r, c] = pv
        out[di]["q"][r, c] = qv
        out[di]["sig"][r, c] = rj
    return out


def plot_pi_difference(pixel_result, homology_dimensions, out_path, title):
    """Heatmap of the PD-vs-control persistence-image difference per dimension,
    with FDR-significant pixels outlined."""
    n_dims = len(homology_dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.6 * n_dims, 4.2),
                             squeeze=False)
    axes = axes.ravel()
    for ax, di, k in zip(axes, range(n_dims), homology_dimensions):
        eff = pixel_result[di]["effect"]
        vlim = np.nanmax(np.abs(eff)) or 1.0
        im = ax.imshow(eff, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       aspect="auto")
        # Outline significant pixels.
        sig = pixel_result[di]["sig"]
        ys, xs = np.where(sig)
        ax.scatter(xs, ys, marker="s", facecolors="none", edgecolors="black",
                   s=40, linewidths=1.0, label="q<0.05")
        ax.set_title(f"H{k}  (disease $-$ control)")
        ax.set_xlabel("birth bin")
        ax.set_ylabel("persistence bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if sig.any():
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Persistence-image difference\n{title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pi_classifier_auc(flat_features, labels, n_splits=5, n_perm=1000, seed=0):
    """Cross-validated PD-vs-control classification on persistence-image
    vectors, with a label-permutation null for the AUC.

    Returns dict: observed cv AUC (mean over folds), permutation p-value,
    per-fold AUCs, and the null distribution mean. Uses a scaled logistic
    regression in a Pipeline (fit inside each fold — no leakage). Falls back
    gracefully if a class is too small to stratify."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = np.asarray(flat_features, float)
    y = np.asarray(labels, int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    min_class = min(n_pos, n_neg)
    if min_class < 3:
        return {"auc": np.nan, "p_perm": np.nan, "fold_aucs": [],
                "null_mean": np.nan, "note": "too few subjects per group"}
    k = min(n_splits, min_class)

    def _cv_auc(yy, rng_seed):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rng_seed)
        aucs = []
        for tr, te in skf.split(X, yy):
            if len(np.unique(yy[te])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"))
            clf.fit(X[tr], yy[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(yy[te], prob))
        return np.array(aucs)

    obs_folds = _cv_auc(y, seed)
    obs = float(np.mean(obs_folds)) if obs_folds.size else np.nan

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        fa = _cv_auc(yp, int(rng.integers(1e9)))
        if fa.size:
            null.append(np.mean(fa))
    null = np.array(null)
    # One-sided: how often does the null reach the observed AUC?
    p_perm = (np.sum(null >= obs) + 1) / (null.size + 1) if null.size else np.nan
    return {"auc": obs, "p_perm": float(p_perm),
            "fold_aucs": [float(a) for a in obs_folds],
            "null_mean": float(np.mean(null)) if null.size else np.nan}


# --------------------------------------------------------------------------- #
# Persistence landscapes: averageable summary + L2 permutation test
# --------------------------------------------------------------------------- #

def persistence_landscapes(diagram_list, homology_dimensions, n_layers=3,
                           n_bins=100):
    """Vectorise diagrams into persistence landscapes.

    Unlike diagrams, landscapes live in a vector space, so a group *mean*
    landscape is well-defined (this is the averageable stand-in for an
    'average barcode'). giotto returns (n_samples, n_dims*n_layers, n_bins)
    with channels ordered dim-major/layer-minor: channel for dim-index ``di``
    and layer ``l`` is ``di*n_layers + l``. Also returns the filtration-axis
    sample points so curves can be plotted against d.
    """
    from gtda.diagrams import PersistenceLandscape
    X = _stack_diagrams(diagram_list, homology_dimensions)
    pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    land = pl.fit_transform(X)  # (n_samples, n_dims*n_layers, n_bins)
    # Recover the sample x-axis from the fitted transformer when available.
    samplings = getattr(pl, "samplings_", None)
    return land, samplings


def _dim_channels(di, n_layers):
    """Channel indices for homology-dimension index ``di`` (dim-major order)."""
    return list(range(di * n_layers, (di + 1) * n_layers))


def landscape_l2_permutation(land_disease, land_control, n_perm=1000, seed=0):
    """Permutation test on the L2 distance between the two group-MEAN
    landscapes, per homology dimension.

    Statistic: || mean_disease - mean_control ||_2 over that dimension's
    channels/bins. Labels are shuffled ``n_perm`` times to build the null.
    Returns {di: {'l2': observed, 'p_perm': p, 'null_mean': ...}}.
    ``land_*`` are (n_subjects, n_channels, n_bins)."""
    d = np.asarray(land_disease, float)
    c = np.asarray(land_control, float)
    n_d, n_ch, n_bins = d.shape
    n_layers_times_dims = n_ch
    pooled = np.concatenate([d, c], axis=0)
    labels = np.array([1] * d.shape[0] + [0] * c.shape[0])
    rng = np.random.default_rng(seed)

    # Infer n_layers from channel count if dims known by caller; here we test
    # per channel-block the caller passes, so operate on the whole array and
    # let the caller slice per dimension before calling. (We test all channels
    # given.) Observed:
    def _l2(mask_a, mask_b):
        ma = pooled[mask_a].mean(axis=0)
        mb = pooled[mask_b].mean(axis=0)
        return float(np.sqrt(np.sum((ma - mb) ** 2)))

    obs = _l2(labels == 1, labels == 0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        null[i] = _l2(perm == 1, perm == 0)
    p_perm = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"l2": obs, "p_perm": float(p_perm),
            "null_mean": float(np.mean(null))}


def plot_group_landscape(land_disease, land_control, di, n_layers, samplings,
                         out_path, title, disease_name, control_name):
    """Group-mean persistence landscape for one homology dimension (layer 0,
    the most prominent, drawn solid; deeper layers faded), PD vs control, with
    a +/- SEM band on layer 0. ``land_*`` are (n_subjects, n_channels, n_bins)
    for this dimension only (already sliced to that dim's channels)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    n_bins = land_disease.shape[2]
    x = (samplings if samplings is not None and len(np.ravel(samplings)) == n_bins
         else np.linspace(0.0, 1.0, n_bins))
    x = np.ravel(x)[:n_bins]

    for land, name, colour in ((land_disease, disease_name, "tab:red"),
                               (land_control, control_name, "tab:blue")):
        if land is None or land.shape[0] == 0:
            continue
        # layer 0 mean + SEM band
        l0 = land[:, 0, :]
        mean0 = np.nanmean(l0, axis=0)
        sem0 = np.nanstd(l0, axis=0, ddof=1) / np.sqrt(max(l0.shape[0], 1))
        ax.plot(x, mean0, color=colour, label=f"{name} (layer 1)")
        ax.fill_between(x, mean0 - sem0, mean0 + sem0, color=colour, alpha=0.2)
        # deeper layers: group means only, faded
        for l in range(1, min(n_layers, land.shape[1])):
            ax.plot(x, np.nanmean(land[:, l, :], axis=0), color=colour,
                    alpha=0.35, linewidth=1)

    ax.set_xlabel(r"Filtration $d = 1 - |w|$")
    ax.set_ylabel("Landscape value $\\lambda(d)$")
    ax.set_title(f"Group-mean persistence landscape (H{di})\n{title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cross-cycle survival: within-subject stride-to-stride topology
# --------------------------------------------------------------------------- #

def _group_cycle_items(items):
    """Group cycle-level item indices by (sub_id, task, kinematic, direction).
    Returns {key: [indices in cycle order]}. Relies on the stem format
    '<sub>_<task>_<kin>_<dir>_cycleNNN'."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(items):
        if it["level"] != "cycle":
            continue
        parts = it["stem"].rsplit("_", 4)  # sub, task, kin, dir, cycleNNN
        direction = it["direction"]
        if len(parts) == 5:
            sub, task, kin, _dir, cyc = parts
        else:
            continue
        groups[(sub, task, kin, direction, it["group"])].append((cyc, i))
    # sort each by cycle label
    ordered = {}
    for key, lst in groups.items():
        lst.sort(key=lambda t: t[0])
        ordered[key] = [i for _, i in lst]
    return ordered


def cross_cycle_betti_counts(cycle_diagrams, homology_dimensions,
                             distance_values, t_index):
    """Betti number of each cycle's diagram at one fixed threshold index.
    Returns {k: array over cycles}."""
    d_at = distance_values[t_index]
    out = {k: [] for k in homology_dimensions}
    for diag in cycle_diagrams:
        b = betti_curve_from_diagram(diag, homology_dimensions, distance_values)
        for k in homology_dimensions:
            out[k].append(int(b[k][t_index]))
    return {k: np.array(v) for k, v in out.items()}, d_at


def cross_cycle_bottleneck(cycle_diagrams, homology_dimensions):
    """Consecutive-cycle bottleneck distances d(cycle_i, cycle_{i+1}).
    Returns the trace (length n_cycles-1) and its mean (a per-subject scalar of
    'topological stride-to-stride variability'). Needs >= 2 cycles."""
    if len(cycle_diagrams) < 2:
        return np.array([]), np.nan
    from gtda.diagrams import PairwiseDistance
    X = _stack_diagrams(list(cycle_diagrams), homology_dimensions)
    D = PairwiseDistance(metric="bottleneck").fit_transform(X)
    trace = np.array([D[i, i + 1] for i in range(D.shape[0] - 1)])
    return trace, float(np.mean(trace))


def plot_cross_cycle(betti_counts, d_at, bottleneck_trace, out_path, title,
                     homology_dimensions):
    """Two-panel per-subject figure: (top) Betti count per cycle at fixed t,
    (bottom) consecutive-cycle bottleneck distance."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)
    ax0, ax1 = axes
    for k in homology_dimensions:
        if k in betti_counts:
            ax0.plot(range(len(betti_counts[k])), betti_counts[k], marker="o",
                     color=DIM_COLORS.get(k), label=f"$b_{{{k}}}$")
    ax0.set_xlabel("gait cycle index")
    ax0.set_ylabel("Betti number")
    ax0.set_title(f"Feature count across cycles at $d={d_at:.2f}$ "
                  f"($|w|\\geq{1 - d_at:.2f}$)")
    ax0.legend(); ax0.grid(True, alpha=0.3)

    if bottleneck_trace.size:
        ax1.plot(range(1, len(bottleneck_trace) + 1), bottleneck_trace,
                 marker="s", color="tab:purple")
        ax1.axhline(np.mean(bottleneck_trace), color="gray", linestyle="--",
                    label=f"mean = {np.mean(bottleneck_trace):.3f}")
        ax1.legend()
    ax1.set_xlabel("consecutive cycle pair (i, i+1)")
    ax1.set_ylabel("bottleneck distance")
    ax1.set_title("Topological stride-to-stride variability")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Group comparison: per-subject features, distribution-aware tests
# --------------------------------------------------------------------------- #

def scalar_features_from_diagram(diagram, homology_dimensions):
    """Reduce one persistence diagram to interpretable scalars per dimension.

    Per homology dimension k returns:
        n_features     — number of (birth, death) points
        total_pers     — sum of lifetimes (death - birth)
        max_pers       — longest lifetime (0.0 if none)
        mean_pers      — mean lifetime (0.0 if none)
        entropy        — persistence entropy of the (normalised) lifetimes

    Essential/infinite classes are kept: their death has already been set to
    the infinity sentinel by FlagserPersistence, so their lifetime is large
    but finite and enters the sums like any other feature. This keeps the
    features comparable across subjects (all diagrams share the sentinel).
    """
    births, deaths, dims = diagram[:, 0], diagram[:, 1], diagram[:, 2]
    feats = {}
    for k in homology_dimensions:
        mask = dims == k
        life = (deaths[mask] - births[mask])
        life = life[life > 0]  # drop zero-persistence / padding rows
        n = int(life.size)
        if n == 0:
            feats[f"H{k}_n_features"] = 0
            feats[f"H{k}_total_pers"] = 0.0
            feats[f"H{k}_max_pers"] = 0.0
            feats[f"H{k}_mean_pers"] = 0.0
            feats[f"H{k}_entropy"] = 0.0
            continue
        total = float(life.sum())
        p = life / total
        entropy = float(-np.sum(p * np.log(p))) if total > 0 else 0.0
        feats[f"H{k}_n_features"] = n
        feats[f"H{k}_total_pers"] = total
        feats[f"H{k}_max_pers"] = float(life.max())
        feats[f"H{k}_mean_pers"] = float(life.mean())
        feats[f"H{k}_entropy"] = entropy
    return feats


def _cohens_d(a, b):
    """Cohen's d with pooled SD (unbiased pooled variance). Positive => group
    a > group b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def _rank_biserial(a, b, u_stat):
    """Rank-biserial correlation from the Mann-Whitney U (group a vs b).
    r = 1 - 2U/(na*nb); positive => group a tends to exceed group b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - (2.0 * u_stat) / (na * nb)


def compare_feature(disease_vals, control_vals, alpha_normality=0.05):
    """Distribution-aware two-sample comparison of one feature.

    - Shapiro-Wilk normality on each group (needs n >= 3).
    - If both groups pass -> Welch's t-test + Cohen's d.
    - Otherwise           -> Mann-Whitney U + rank-biserial correlation.

    Effect direction is disease - control (positive => disease larger).
    Returns a dict of statistics.
    """
    from scipy import stats

    d = np.asarray(disease_vals, float)
    d = d[~np.isnan(d)]
    c = np.asarray(control_vals, float)
    c = c[~np.isnan(c)]

    res = {
        "n_disease": int(d.size), "n_control": int(c.size),
        "mean_disease": float(d.mean()) if d.size else np.nan,
        "mean_control": float(c.mean()) if c.size else np.nan,
        "median_disease": float(np.median(d)) if d.size else np.nan,
        "median_control": float(np.median(c)) if c.size else np.nan,
        "test": None, "stat": np.nan, "p": np.nan,
        "effect_name": None, "effect": np.nan, "normal": None,
    }
    if d.size < 3 or c.size < 3:
        res["test"] = "insufficient_n"
        return res

    # Constant input breaks Shapiro; treat as non-normal.
    def _is_normal(x):
        if np.allclose(x, x[0]):
            return False
        try:
            return stats.shapiro(x).pvalue > alpha_normality
        except Exception:
            return False

    both_normal = _is_normal(d) and _is_normal(c)
    res["normal"] = bool(both_normal)

    if both_normal:
        t, p = stats.ttest_ind(d, c, equal_var=False)  # Welch
        res.update(test="welch_t", stat=float(t), p=float(p),
                   effect_name="cohens_d", effect=_cohens_d(d, c))
    else:
        u, p = stats.mannwhitneyu(d, c, alternative="two-sided")
        res.update(test="mann_whitney_u", stat=float(u), p=float(p),
                   effect_name="rank_biserial",
                   effect=_rank_biserial(d, c, u))
    return res


def _fdr_bh(pvals):
    """Benjamini-Hochberg FDR. Returns (rejected, qvalues) aligned to input.
    NaN p-values are passed through as NaN and excluded from the correction."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    rej = np.zeros(p.shape, bool)
    finite = np.where(~np.isnan(p))[0]
    if finite.size == 0:
        return rej, q
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    qv = ranked * m / (np.arange(1, m + 1))
    qv = np.minimum.accumulate(qv[::-1])[::-1]  # enforce monotonicity
    qv = np.clip(qv, 0, 1)
    q_full = np.empty(m)
    q_full[order] = qv
    q[finite] = q_full
    rej[finite] = q_full < 0.05
    return rej, q


def plot_group_betti_curves(betti_disease, betti_control, filter_thresholds,
                            dim, out_path, title, sig_mask=None,
                            disease_name="Disease", control_name="Control"):
    """Group-mean Betti curves (mean +/- SEM) for one homology dimension.
    ``betti_*`` are (n_subjects, n_thresholds) arrays. ``sig_mask`` (optional,
    length n_thresholds) marks thresholds with a significant group difference.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    def _mean_sem(mat):
        mat = np.asarray(mat, float)
        mean = np.nanmean(mat, axis=0)
        n = np.sum(~np.isnan(mat), axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        return mean, sem

    for mat, name, colour in ((betti_disease, disease_name, "tab:red"),
                              (betti_control, control_name, "tab:blue")):
        if mat is None or len(mat) == 0:
            continue
        mean, sem = _mean_sem(mat)
        ax.plot(filter_thresholds, mean, marker="o", color=colour, label=name)
        ax.fill_between(filter_thresholds, mean - sem, mean + sem,
                        color=colour, alpha=0.2)

    if sig_mask is not None and np.any(sig_mask):
        ymax = ax.get_ylim()[1]
        ax.scatter(np.asarray(filter_thresholds)[sig_mask],
                   np.full(int(np.sum(sig_mask)), ymax * 0.98),
                   marker="*", color="black", s=60, zorder=5,
                   label="q < 0.05")

    ax.set_xlabel(r"Filter threshold $t$  (edges kept when $|w_{ij}| \geq t$)")
    ax.set_ylabel(f"Betti number $b_{{{dim}}}$")
    ax.set_title(f"Group-mean Betti curve $b_{{{dim}}}$\n{title}")
    ax.set_xticks(filter_thresholds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_group_comparison(items, diagrams, homology_dimensions,
                         filter_thresholds, distance_values, out_root,
                         disease_name, control_name, pi_n_bins=20,
                         pi_n_perm=1000, cross_cycle_scalars=None,
                         pl_n_layers=3, pl_n_bins=100, pl_n_perm=1000):
    """Subject-level group comparison of TDA features + Betti curves.

    Only subject-level items are used for the scalar/Betti/PI tests (cycles are
    not independent across subjects and must not be pooled). Comparison is run
    separately per (task, kinematic, direction) stratum so unlike graphs are
    never mixed. Per stratum it writes:
      - scalar-feature statistics (into the combined CSV),
      - a group-mean Betti-curve figure per dimension,
      - a persistence-image difference heatmap (pixel-wise test, FDR) + a
        cross-validated classifier AUC with a permutation p-value.

    ``cross_cycle_scalars`` (optional): dict
        {(task, kinematic, direction): {'disease': [means], 'control': [means]}}
        of per-subject mean consecutive-cycle bottleneck distances. When given,
        each stratum's topological stride-to-stride variability is added to the
        combined test table as feature 'cross_cycle_bottleneck_mean'.
    """
    from collections import defaultdict

    # Keep only subject-level graphs, carrying their group + stratum keys.
    sub_idx = [i for i, it in enumerate(items) if it["level"] == "subject"]
    if not sub_idx:
        print("\nNo subject-level graphs — skipping group comparison "
              "(cycle-level data are not pooled across subjects).")
        return

    # Parse stratum (task, kinematic, direction) back out of each item. The
    # stem is '<sub>_<task>_<kinematic>_<dir>_mean'; direction is stored
    # explicitly, and subdir encodes the group.
    strata = defaultdict(list)  # (task, kinematic, direction) -> list of idx
    for i in sub_idx:
        it = items[i]
        # stem: sub_task_kinematic_dir_mean -> split carefully from the right
        parts = it["stem"].rsplit("_", 4)
        # parts = [sub_id, task, kinematic, dir, 'mean'] when sub_id has no '_'
        # Fall back to stored direction to stay robust to underscores in ids.
        direction = it["direction"]
        if len(parts) == 5:
            _, task, kinematic, _dir, _ = parts
        else:
            task, kinematic = "task", "kin"
        strata[(task, kinematic, direction)].append(i)

    stats_dir = out_root / "group_comparison"
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for (task, kinematic, direction), idxs in sorted(strata.items()):
        # Split indices by group.
        dis_idx = [i for i in idxs if items[i]["group"] == disease_name]
        ctl_idx = [i for i in idxs if items[i]["group"] == control_name]
        if not dis_idx or not ctl_idx:
            print(f"  [{task}/{kinematic}/{direction}] missing one group "
                  f"(disease={len(dis_idx)}, control={len(ctl_idx)}) — skipped.")
            continue

        # ---- scalar features per subject ----
        dis_feats = [scalar_features_from_diagram(diagrams[i], homology_dimensions)
                     for i in dis_idx]
        ctl_feats = [scalar_features_from_diagram(diagrams[i], homology_dimensions)
                     for i in ctl_idx]
        feature_names = list(dis_feats[0].keys())

        rows, pvals = [], []
        for fname in feature_names:
            dvals = [f[fname] for f in dis_feats]
            cvals = [f[fname] for f in ctl_feats]
            r = compare_feature(dvals, cvals)
            r.update(task=task, kinematic=kinematic, direction=direction,
                     feature=fname)
            rows.append(r)
            pvals.append(r["p"])

        # FDR across all features within this stratum.
        rejected, qvals = _fdr_bh(pvals)
        for r, q, rej in zip(rows, qvals, rejected):
            r["q_fdr"] = float(q) if not np.isnan(q) else np.nan
            r["significant_fdr"] = bool(rej)
        all_rows.extend(rows)

        # ---- group-mean Betti curves per dimension ----
        for k in homology_dimensions:
            b_dis = np.stack([
                betti_curve_from_diagram(diagrams[i], homology_dimensions,
                                         distance_values)[k]
                for i in dis_idx])
            b_ctl = np.stack([
                betti_curve_from_diagram(diagrams[i], homology_dimensions,
                                         distance_values)[k]
                for i in ctl_idx])

            # Pointwise test across thresholds + FDR over thresholds.
            pw_p = []
            for ti in range(len(filter_thresholds)):
                rr = compare_feature(b_dis[:, ti], b_ctl[:, ti])
                pw_p.append(rr["p"])
            _, pw_q = _fdr_bh(pw_p)
            sig_mask = np.array([(not np.isnan(q)) and q < 0.05 for q in pw_q])

            title = f"{task} | {kinematic} | {direction}"
            out_png = (stats_dir /
                       f"betti_group_{task}_{kinematic}_{direction}_H{k}.png")
            plot_group_betti_curves(
                b_dis, b_ctl, filter_thresholds, k, out_png, title,
                sig_mask=sig_mask, disease_name=disease_name,
                control_name=control_name)

        # ---- persistence images: pixel-wise heatmap + classifier ----
        strat_title = f"{task} | {kinematic} | {direction}"
        try:
            dis_diags = [diagrams[i] for i in dis_idx]
            ctl_diags = [diagrams[i] for i in ctl_idx]
            img_dis, flat_dis = persistence_images(
                dis_diags, homology_dimensions, n_bins=pi_n_bins)
            img_ctl, flat_ctl = persistence_images(
                ctl_diags, homology_dimensions, n_bins=pi_n_bins)

            # giotto fits bins per call, so dis/ctl images live on different
            # grids. Refit one PI on the pooled set for a shared grid.
            all_diags = dis_diags + ctl_diags
            img_all, flat_all = persistence_images(
                all_diags, homology_dimensions, n_bins=pi_n_bins)
            n_d = len(dis_diags)
            img_dis, img_ctl = img_all[:n_d], img_all[n_d:]
            flat_all_labels = np.array([1] * n_d + [0] * len(ctl_diags))

            pixel_res = pi_pixelwise_difference(img_dis, img_ctl)
            plot_pi_difference(
                pixel_res, homology_dimensions,
                stats_dir / f"pi_diff_{task}_{kinematic}_{direction}.png",
                strat_title)
            n_sig_pix = sum(int(pixel_res[di]["sig"].sum())
                            for di in range(len(homology_dimensions)))

            clf = pi_classifier_auc(flat_all, flat_all_labels,
                                    n_perm=pi_n_perm)
            all_rows.append({
                "task": task, "kinematic": kinematic, "direction": direction,
                "feature": "persistence_image_classifier_AUC",
                "test": "logreg_cv_permutation", "normal": None,
                "n_disease": n_d, "n_control": len(ctl_diags),
                "mean_disease": np.nan, "mean_control": np.nan,
                "median_disease": np.nan, "median_control": np.nan,
                "stat": clf["auc"], "p": clf["p_perm"],
                "q_fdr": np.nan,  # standalone; not part of the feature FDR set
                "significant_fdr": (not np.isnan(clf["p_perm"]))
                                   and clf["p_perm"] < 0.05,
                "effect_name": "cv_AUC", "effect": clf["auc"],
            })
            print(f"  [{strat_title}] persistence image: "
                  f"{n_sig_pix} pixel(s) FDR-sig; "
                  f"classifier AUC={clf['auc']:.3f} "
                  f"(perm p={clf['p_perm']}).")
        except Exception as e:  # keep the pipeline running on PI failure
            print(f"  [{strat_title}] persistence-image step skipped: {e}")

        # ---- persistence landscapes: group-mean figure + L2 perm test ----
        try:
            dis_diags = [diagrams[i] for i in dis_idx]
            ctl_diags = [diagrams[i] for i in ctl_idx]
            n_d = len(dis_diags)
            # One landscape fit on the pooled set -> shared sampling grid.
            land_all, samplings = persistence_landscapes(
                dis_diags + ctl_diags, homology_dimensions,
                n_layers=pl_n_layers, n_bins=pl_n_bins)
            land_dis, land_ctl = land_all[:n_d], land_all[n_d:]

            for di, k in enumerate(homology_dimensions):
                ch = _dim_channels(di, pl_n_layers)
                ld = land_dis[:, ch, :]
                lc = land_ctl[:, ch, :]
                perm = landscape_l2_permutation(ld, lc, n_perm=pl_n_perm)
                plot_group_landscape(
                    ld, lc, k, pl_n_layers, samplings,
                    stats_dir / f"landscape_group_{task}_{kinematic}_{direction}_H{k}.png",
                    strat_title, disease_name, control_name)
                all_rows.append({
                    "task": task, "kinematic": kinematic, "direction": direction,
                    "feature": f"H{k}_landscape_L2",
                    "test": "landscape_l2_permutation", "normal": None,
                    "n_disease": n_d, "n_control": len(ctl_diags),
                    "mean_disease": np.nan, "mean_control": np.nan,
                    "median_disease": np.nan, "median_control": np.nan,
                    "stat": perm["l2"], "p": perm["p_perm"],
                    "q_fdr": np.nan,  # standalone permutation test
                    "significant_fdr": (not np.isnan(perm["p_perm"]))
                                       and perm["p_perm"] < 0.05,
                    "effect_name": "landscape_L2", "effect": perm["l2"],
                })
                print(f"  [{strat_title}] H{k} landscape L2={perm['l2']:.4g} "
                      f"(perm p={perm['p_perm']}).")
        except Exception as e:
            print(f"  [{strat_title}] persistence-landscape step skipped: {e}")

        # ---- cross-cycle bottleneck variability rollup (if provided) ----
        if cross_cycle_scalars and (task, kinematic, direction) in cross_cycle_scalars:
            cc = cross_cycle_scalars[(task, kinematic, direction)]
            r = compare_feature(cc.get("disease", []), cc.get("control", []))
            r.update(task=task, kinematic=kinematic, direction=direction,
                     feature="cross_cycle_bottleneck_mean")
            # single test -> q mirrors p (not part of the feature FDR family)
            r["q_fdr"] = r["p"]
            r["significant_fdr"] = (not np.isnan(r["p"])) and r["p"] < 0.05
            all_rows.append(r)

    if not all_rows:
        print("\nGroup comparison produced no rows (no stratum had both groups).")
        return

    # ---- write the combined stats table (CSV + pickle) ----
    import pandas as pd
    df = pd.DataFrame(all_rows)
    col_order = ["task", "kinematic", "direction", "feature", "test", "normal",
                 "n_disease", "n_control", "mean_disease", "mean_control",
                 "median_disease", "median_control", "stat", "p", "q_fdr",
                 "significant_fdr", "effect_name", "effect"]
    df = df[[c for c in col_order if c in df.columns]]
    csv_path = stats_dir / "tda_group_comparison.csv"
    pkl_path = stats_dir / "tda_group_comparison.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    n_sig = int(df["significant_fdr"].sum())
    print(f"\nGroup comparison written to {csv_path}")
    print(f"  {len(df)} feature test(s) across {len(strata)} stratum/strata; "
          f"{n_sig} significant after FDR.")
    if n_sig:
        sig = df[df["significant_fdr"]]
        for _, r in sig.iterrows():
            print(f"    * {r['task']}/{r['kinematic']}/{r['direction']} "
                  f"{r['feature']}: {r['test']} p={r['p']:.4g} "
                  f"q={r['q_fdr']:.4g} {r['effect_name']}={r['effect']:.3f}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def tda_main(diagnosis, kinematics_list, task_names, tracking_systems, runs,
             pd_on, base_path, marker_list_affect, result_base_path, full,
             correlation_method):
    """Run persistent-homology TDA on kinectomes; signature mirrors the other
    *_main entry points. All TDA parameters are read from config."""
    from config import (TDA_LEVELS, TDA_HOMOLOGY_DIMENSIONS,
                        TDA_THRESHOLD_START, TDA_THRESHOLD_STOP,
                        TDA_THRESHOLD_STEP, TDA_INFINITY_SENTINEL)

    filter_thresholds = np.arange(TDA_THRESHOLD_START,
                                  TDA_THRESHOLD_STOP + 1e-9,
                                  TDA_THRESHOLD_STEP)
    distance_values = 1.0 - filter_thresholds

    out_root = Path(result_base_path) / "tda"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("TOPOLOGICAL DATA ANALYSIS (persistent homology)")
    print(f"Type: {'FULL' if full else 'DIRECTIONAL'} | levels: {TDA_LEVELS} | "
          f"dims: {TDA_HOMOLOGY_DIMENSIONS}")
    print(f"Thresholds t: {list(np.round(filter_thresholds, 3))}")
    print(f"Output: {out_root}")
    print("=" * 64)

    print("\nGathering kinectomes (matched groups, marker exclusion applied)...")
    items = _gather_matrices(diagnosis, kinematics_list, task_names,
                             tracking_systems, runs, pd_on, full,
                             correlation_method, marker_list_affect, TDA_LEVELS)
    if not items:
        print("No kinectomes found to analyse. Check KINECTOME_SAVE_PATH and TASK_NAMES.")
        return

    n_cycle = sum(1 for it in items if it['level'] == 'cycle')
    n_subject = sum(1 for it in items if it['level'] == 'subject')
    print(f"  {len(items)} graph(s) to analyse "
          f"({n_subject} subject-level, {n_cycle} cycle-level).")

    print("\nComputing persistence diagrams with FlagserPersistence "
          "(dense flag complexes can be slow)...")
    matrices = [it['matrix'] for it in items]
    diagrams = compute_persistence_diagrams(matrices, TDA_HOMOLOGY_DIMENSIONS,
                                            TDA_INFINITY_SENTINEL)

    progress = tqdm(list(zip(items, diagrams)), desc="Plotting", unit="graph")
    for it, diagram in progress:
        progress.set_description(f"Plotting {it['stem']}")
        out_dir = out_root / it['subdir']
        out_dir.mkdir(parents=True, exist_ok=True)
        betti = betti_curve_from_diagram(diagram, TDA_HOMOLOGY_DIMENSIONS, distance_values)
        plot_betti_curves(betti, filter_thresholds,
                          out_dir / f"{it['stem']}_betti_curve.png", it['stem'])
        plot_persistence_diagrams(diagram, TDA_HOMOLOGY_DIMENSIONS,
                                  TDA_INFINITY_SENTINEL,
                                  out_dir / f"{it['stem']}_persistence_diagram.png",
                                  it['stem'])
        plot_barcode(diagram, TDA_HOMOLOGY_DIMENSIONS, TDA_INFINITY_SENTINEL,
                     out_dir / f"{it['stem']}_barcode.png", it['stem'])

    print(f"\nPer-graph plots saved under {out_root}")
    print("  subject-level: one diagram per subject (intra-subject topology).")
    print("  cycle-level:   one diagram per gait cycle (inter-cycle topology).")

    disease_name = f"{diagnosis[0][10:].capitalize()}"
    control_name = "Control"

    # -------------------------------------------------------------------- #
    # Cross-cycle survival (within-subject stride-to-stride topology).
    # Per subject: Betti-count-per-cycle strip at a fixed threshold + the
    # consecutive-cycle bottleneck-distance trace. The mean bottleneck
    # distance is a per-subject scalar ("topological gait variability") that
    # is rolled up into the group comparison below. Needs cycle-level data.
    # -------------------------------------------------------------------- #
    cross_cycle_scalars = None
    cyc_groups = _group_cycle_items(items)
    if cyc_groups:
        from collections import defaultdict
        # Fixed threshold for the count strip: the middle of the range.
        t_index = len(filter_thresholds) // 2
        cc_dir = out_root / "cross_cycle"
        cc_dir.mkdir(parents=True, exist_ok=True)
        cross_cycle_scalars = defaultdict(lambda: {"disease": [], "control": []})

        print("\n" + "=" * 64)
        print("CROSS-CYCLE SURVIVAL (within-subject, stride-to-stride)")
        print(f"  fixed threshold for count strip: t="
              f"{filter_thresholds[t_index]:.3f}")
        print("=" * 64)

        cc_progress = tqdm(sorted(cyc_groups.items()),
                           desc="Cross-cycle", unit="subject")
        for (sub, task, kin, direction, group), idx_list in cc_progress:
            cyc_diags = [diagrams[i] for i in idx_list]
            if len(cyc_diags) < 2:
                continue
            counts, d_at = cross_cycle_betti_counts(
                cyc_diags, TDA_HOMOLOGY_DIMENSIONS, distance_values, t_index)
            trace, mean_bn = cross_cycle_bottleneck(
                cyc_diags, TDA_HOMOLOGY_DIMENSIONS)

            title = f"{sub} | {task} | {kin} | {direction} ({group})"
            plot_cross_cycle(
                counts, d_at, trace,
                cc_dir / f"{sub}_{task}_{kin}_{direction}_crosscycle.png",
                title, TDA_HOMOLOGY_DIMENSIONS)

            key = (task, kin, direction)
            bucket = "disease" if group == disease_name else "control"
            if not np.isnan(mean_bn):
                cross_cycle_scalars[key][bucket].append(mean_bn)

        cross_cycle_scalars = {k: v for k, v in cross_cycle_scalars.items()}
        print(f"  per-subject cross-cycle plots saved under {cc_dir}")

    # -------------------------------------------------------------------- #
    # Subject-level group comparison: scalar features + group-mean Betti
    # curves + persistence-image heatmap/classifier, plus the cross-cycle
    # variability rollup. Skipped automatically if 'subject' not in levels.
    # -------------------------------------------------------------------- #
    print("\n" + "=" * 64)
    print("GROUP COMPARISON (subject level, distribution-aware)")
    print("  normal (Shapiro) -> Welch t-test + Cohen's d")
    print("  non-normal       -> Mann-Whitney U + rank-biserial r")
    print("  FDR: Benjamini-Hochberg within each task/kinematic/direction")
    print("=" * 64)
    run_group_comparison(
        items, diagrams, TDA_HOMOLOGY_DIMENSIONS, filter_thresholds,
        distance_values, out_root, disease_name, control_name,
        cross_cycle_scalars=cross_cycle_scalars,
    )