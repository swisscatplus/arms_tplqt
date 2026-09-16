"""Motion-capture to world calibrations, one per calibration session.

Spatula poses are recorded in the motion-capture frame while the vial pose is
measured in the world frame of the workbench, so every demonstration has to be
lifted into world with the calibration that was in force when it was recorded.
The calibrations are kept in ``calibrations.json`` next to this module, together
with the mapping from dataset directory name to calibration date.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations.json")

Calibration = Tuple[np.ndarray, np.ndarray]


def load_config(path: Optional[str] = None) -> dict:
    """Parse the calibration file (the packaged one unless ``path`` is given)."""
    with open(path or CONFIG_PATH) as f:
        return json.load(f)


_CONFIG = load_config()


def calibration(name: str, config: Optional[dict] = None) -> Calibration:
    """``(translation, quaternion)`` of the session called ``name``.

    The pair is the pose of the motion-capture frame in world, so
    ``p_world = R(quaternion) p_mocap + translation`` with the quaternion in
    xyzw order.
    """
    config = config or _CONFIG
    try:
        entry = config["calibrations"][name]
    except KeyError:
        known = ", ".join(sorted(config["calibrations"]))
        raise KeyError(f"unknown calibration {name!r}; known calibrations: {known}") from None
    return np.asarray(entry["translation"], float), np.asarray(entry["quaternion"], float)


def calibration_names(config: Optional[dict] = None) -> Tuple[str, ...]:
    """Names of every calibration in the file, sorted."""
    config = config or _CONFIG
    return tuple(sorted(config["calibrations"]))


def registry(config: Optional[dict] = None) -> Dict[str, Calibration]:
    """``{name: (translation, quaternion)}`` for every calibration."""
    config = config or _CONFIG
    return {name: calibration(name, config) for name in config["calibrations"]}


def calibration_name_for_dataset(dataset: str, config: Optional[dict] = None) -> str:
    """Calibration a dataset was recorded under, looked up by directory name."""
    config = config or _CONFIG
    key = os.path.basename(str(dataset).rstrip("/\\"))
    try:
        return config["datasets"][key]
    except KeyError:
        known = ", ".join(sorted(config["datasets"]))
        raise KeyError(
            f"dataset {key!r} has no calibration; add it to {CONFIG_PATH} or pass an "
            f"explicit calibration. Known datasets: {known}") from None


def calibration_for_dataset(dataset: str, config: Optional[dict] = None) -> Calibration:
    """``(translation, quaternion)`` for the dataset's calibration session."""
    return calibration(calibration_name_for_dataset(dataset, config), config)
