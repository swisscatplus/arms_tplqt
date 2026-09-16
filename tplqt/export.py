"""Writing a generated stroke to disk.

The trajectory is stored as a ``.npz`` of plain arrays plus a metadata string, so
it loads without ``allow_pickle`` and can be read by anything that reads numpy
archives.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from .synthesize import Synthesis

FIELDS = ("time", "position", "orientation", "velocity", "angular_velocity")


def save_trajectory(path: str, synthesis: Synthesis, *,
                    metadata: Optional[dict] = None) -> str:
    """Write a generated stroke to ``path``.

    The archive holds the timestamps, the tip position, the spatula orientation as
    xyzw quaternions, the tip velocity and the angular velocity, all in the world
    frame and SI units, plus a JSON ``metadata`` entry describing how the stroke
    was generated. Callers should put the vial pose the stroke was generated for in
    that metadata, since nothing else in the file says where the world frame is
    anchored.
    """
    if synthesis.quat_world is None:
        raise ValueError("this stroke has no orientation; fit the model with orientation=True")
    info = {
        "frame": "world",
        "quaternion_order": "xyzw",
        "units": {"position": "m", "velocity": "m/s", "angular_velocity": "rad/s",
                  "time": "s"},
        "dt": float(synthesis.dt),
        "constrained": bool(synthesis.constrained),
        "solver_status": synthesis.solver_status,
        "jerk_rms": float(synthesis.jerk_rms),
        "tracking_rms": float(synthesis.tracking_rms),
    }
    info.update(metadata or {})
    np.savez(
        path,
        time=np.asarray(synthesis.time, float),
        position=np.asarray(synthesis.pos_world, float),
        orientation=np.asarray(synthesis.quat_world, float),
        velocity=np.asarray(synthesis.vel_world, float),
        angular_velocity=np.asarray(synthesis.omega_world, float),
        metadata=np.array(json.dumps(info)),
    )
    return path if path.endswith(".npz") else path + ".npz"


def load_trajectory(path: str):
    """Read a trajectory written by :func:`save_trajectory`.

    Returns ``(arrays, metadata)``, where ``arrays`` maps each of :data:`FIELDS`
    to its array.
    """
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in FIELDS}
        metadata = json.loads(str(archive["metadata"].item()))
    return arrays, metadata
