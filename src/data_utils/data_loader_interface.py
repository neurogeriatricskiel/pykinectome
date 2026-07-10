"""
data_loader_interface.py — Abstract base class for data loading.
================================================================

To use pykinectome with your own data structure, create a subclass of
``DataLoader`` and implement the two abstract methods below.  Then point
``config.py`` at your class by setting ``DATA_LOADER_CLASS``.

Example
-------
In ``config.py``::

    from src.data_utils.my_loader import MyDataLoader
    DATA_LOADER_CLASS = MyDataLoader

Your loader class (``src/data_utils/my_loader.py``)::

    from src.data_utils.data_loader_interface import DataLoader
    import pandas as pd

    class MyDataLoader(DataLoader):

        def load_raw_data(self, sub_id, task_name, tracksys, run):
            # Load however your data is structured and return a DataFrame
            path = self.base_path / sub_id / f"{task_name}.tsv"
            return pd.read_csv(path, sep='\\t')

        def get_trim_indices(self, data, sub_id, task_name, run):
            # Return (start_frame, stop_frame) as integers.
            # If your data is already trimmed, return (0, len(data)).
            return 0, len(data)

The contract
------------
``load_raw_data`` must return a ``pd.DataFrame`` where:
  - Rows are time samples at the sampling rate defined in ``config.FS``.
  - Columns follow the naming convention ``<marker>_POS_<x|y|z>``.
  - NaN is used for missing/occluded samples.

``get_trim_indices`` must return a tuple ``(start_frame, stop_frame)`` of
integer frame indices (0-based) defining the active walking window within the
raw data.  The pipeline crops the data to ``data[start:stop]`` before
preprocessing.

Both methods receive ``sub_id``, ``task_name``, ``tracksys``, and ``run``
so they can construct whatever file paths or queries your data format requires.
If your data has no concept of ``tracksys`` or ``run``, simply ignore those
parameters in your implementation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class DataLoader(ABC):
    """Abstract base class defining the data loading interface.

    Subclass this and implement :meth:`load_raw_data` and
    :meth:`get_trim_indices` to plug your own data format into the pipeline.

    Parameters
    ----------
    base_path : Path or str
        Root directory of your dataset.  Passed in automatically from
        ``config.BASE_PATH``.
    raw_data_path : Path or str
        Root directory of your raw data files.  Passed in automatically from
        ``config.RAW_DATA_PATH``.
    """

    def __init__(self, base_path: Path, raw_data_path: Path):
        self.base_path = Path(base_path)
        self.raw_data_path = Path(raw_data_path)

    @abstractmethod
    def load_raw_data(self, sub_id: str, task_name: str,
                      tracksys: str, run: str) -> pd.DataFrame:
        """Load raw motion capture data for one subject/task/run.

        Parameters
        ----------
        sub_id : str
            Subject identifier (e.g. ``'pp065'``).
        task_name : str
            Task name (e.g. ``'walkStroop'``).
        tracksys : str
            Tracking system identifier (e.g. ``'omc'``).
        run : str or None
            Run condition (``'on'``, ``'off'``, or ``None`` for controls).

        Returns
        -------
        pd.DataFrame or None
            Raw motion data with shape ``(n_frames, n_channels)``, or
            ``None`` if no data is available for this combination.
        """
        ...


