"""Pickle IO helpers.

Isaac Lab 2.0.2 (which WheeledLab targets) exposed `dump_pickle`/`load_pickle`
from `isaaclab.utils.io`, but Isaac Lab >=2.1 removed them (only yaml +
torchscript helpers remain). WheeledLab needs pickle for RunConfig objects
(which contain slices that yaml can't round-trip), so we re-provide the two
functions here. Lets WheeledLab run on Isaac Lab 2.3.x without editing the
framework. Behaviour matches the original isaaclab implementation.
"""

from __future__ import annotations

import os
import pickle
from typing import Any


def dump_pickle(filename: str, data: Any) -> None:
    """Pickle `data` to `filename` (appends .pkl, makes parent dirs)."""
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filename: str) -> Any:
    """Load a pickled object from `filename`."""
    with open(filename, "rb") as f:
        return pickle.load(f)
