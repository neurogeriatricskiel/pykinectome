"""
bids_data_loader.py — BIDS-structured OMC data loader.
=======================================================

Concrete implementation of :class:`DataLoader` for the Keep Control dataset
(BIDS-formatted optical motion capture data).

This is the default loader used when ``DATA_LOADER_CLASS`` is not set in
``config.py``.  It navigates the BIDS folder structure
``rawdata/sub-<id>/motion/`` and loads the appropriate motion TSV file based
on subject ID, task name, tracking system, and run condition.

For a different data structure, subclass :class:`DataLoader` from
``data_loader_interface.py`` and implement :meth:`load_raw_data`.
"""

from pathlib import Path
import pandas as pd
from src.data_utils.data_loader_interface import DataLoader


class BIDSDataLoader(DataLoader):
    """Load motion capture data from a BIDS-formatted directory tree.

    Expects the following folder structure::

        raw_data_path/
        └── sub-<sub_id>/
            └── motion/
                └── sub-<sub_id>_task-<task>_[run-<run>_]tracksys-<sys>_motion.tsv

    The ``run-<on|off>`` token is present only for PD participants.
    Controls have no run token in their filenames.

    Parameters
    ----------
    base_path : Path or str
        Root of the dataset (parent of ``rawdata/`` and ``derived_data/``).
    raw_data_path : Path or str
        Path to the ``rawdata/`` directory.
    """

    def load_raw_data(self, sub_id: str, task_name: str,
                      tracksys: str, run: str) -> pd.DataFrame | None:
        """Load the motion TSV for one subject/task/tracksys/run.

        Searches ``raw_data_path/sub-<sub_id>/motion/`` for a file whose name
        contains all of ``sub_id``, ``task_name``, ``tracksys``, and
        ``'motion'``.  Among matching files, prefers the one with
        ``run-<run>`` in the name; falls back to files without any run token
        (for controls).

        Parameters
        ----------
        sub_id : str
            Subject identifier (e.g. ``'pp065'``).
        task_name : str
            Task name (e.g. ``'walkStroop'``).
        tracksys : str
            Tracking system (e.g. ``'omc'``).
        run : str or None
            Run condition (``'on'``, ``'off'``, or ``None``).

        Returns
        -------
        pd.DataFrame or None
            Raw motion data, or ``None`` if no matching file is found.
        """
        motion_dir = self.raw_data_path / f'sub-{sub_id}' / 'motion'

        if not motion_dir.exists():
            print(f"  Motion directory not found: {motion_dir}")
            return None

        for file in motion_dir.iterdir():
            fname = file.name
            if (sub_id in fname and task_name in fname
                    and tracksys in fname and 'motion' in fname):
                if run and f'run-{run}' in fname:
                    return pd.read_csv(file, sep='\t', header=0)
                elif not any(f'run-{c}' in fname for c in ['on', 'off']):
                    return pd.read_csv(file, sep='\t', header=0)

        print(f"  No motion file found for sub-{sub_id}, task-{task_name}, "
              f"tracksys-{tracksys}, run-{run}")
        return None
