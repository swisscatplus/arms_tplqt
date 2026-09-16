"""Loading recorded demonstrations from disk.

A dataset is a directory of demonstration folders; a folder is a demonstration
when it holds the four recorded files:

``metadata.json``
    demonstration name, the time of first contact with the material, the spatula
    pose at that instant, the sampled mass and the material descriptors;
``spatula_pose_in_mocap_frame.jsonl``
    the spatula pose stream in the motion-capture frame, one JSON object per
    sample;
``vial_lip_in_world.jsonl``
    the vial-lip pose stream in the world frame, static within a recording;
``force_torque_data_in_bota_frame.jsonl``
    the wrench stream from the sensor between the spatula and its holder. The model
    does not use it, but it is part of the recording and is carried through to the
    prepared stroke, windowed and downsampled alongside the pose, for analysing what
    the spatula felt.

The three streams share a clock and are sampled row for row. Quaternions are
xyzw throughout, matching both the recordings and ``scipy.spatial.transform``.
This module applies no transforms: it returns what was recorded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

REQUIRED_FILES = ("metadata.json", "spatula_pose_in_mocap_frame.jsonl",
                  "vial_lip_in_world.jsonl", "force_torque_data_in_bota_frame.jsonl")

# Largest disagreement allowed between the three streams' timestamps.
CLOCK_TOL_S = 1e-6


@dataclass
class Demonstration:
    """One recorded demonstration, as it was stored.

    The spatula rigid body is calibrated with its origin at the tool tip, so
    ``spatula_pos`` is already the tip trajectory.
    """

    name: str
    dataset: str                    # directory the demonstration was loaded from

    time: np.ndarray                # (N,) seconds, shared by all three streams
    spatula_pos: np.ndarray         # (N, 3) tip position in the motion-capture frame
    spatula_quat: np.ndarray        # (N, 4) xyzw in the motion-capture frame
    wrench: np.ndarray              # (N, 6) force and torque in the sensor frame

    vial_lip_pos: np.ndarray        # (3,) in world
    vial_lip_quat: np.ndarray       # (4,) xyzw in world; its z axis is the vial bore

    contact_time: float             # instant the spatula first touched the material
    contact_pos_mocap: np.ndarray   # (3,) tip position at that instant
    contact_quat_mocap: np.ndarray  # (4,) xyzw spatula orientation at that instant

    sampled_mass_mg: float
    descriptors: dict
    rate_hz: float                  # rate the recording was configured at

    @property
    def n_samples(self) -> int:
        return int(self.time.shape[0])


def is_demonstration(path: str) -> bool:
    """True when ``path`` is a directory holding all of :data:`REQUIRED_FILES`."""
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f)) for f in REQUIRED_FILES)


def _read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _xyz(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], float)


def _xyzw(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"], d["w"]], float)


def load_demonstration(folder: str, *, static_tol_m: float = 1e-3,
                       static_tol_deg: float = 1.0) -> Demonstration:
    """Read one demonstration folder.

    Two recording assumptions are checked rather than assumed: the three streams
    are on a common clock, and the vial neither moves nor turns during a recording.
    Both are load-bearing downstream, so a violation is an error, not a silent
    misalignment.
    """
    folder = os.path.abspath(folder)
    name = os.path.basename(folder.rstrip("/"))
    dataset = os.path.basename(os.path.dirname(folder))

    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    missing = [k for k in ("contact_pose", "contact_time", "scoop_weight_mg") if k not in meta]
    if missing:
        raise ValueError(f"{name}: metadata.json is missing {missing}")

    spatula = _read_jsonl(os.path.join(folder, "spatula_pose_in_mocap_frame.jsonl"))
    vial = _read_jsonl(os.path.join(folder, "vial_lip_in_world.jsonl"))
    wrench = _read_jsonl(os.path.join(folder, "force_torque_data_in_bota_frame.jsonl"))

    t_spatula = np.array([r["time"] for r in spatula])
    t_vial = np.array([r["time"] for r in vial])
    t_wrench = np.array([r["time"] for r in wrench])
    if not (len(t_spatula) == len(t_vial) == len(t_wrench)):
        raise ValueError(f"{name}: streams have different lengths "
                         f"({len(t_spatula)}, {len(t_vial)}, {len(t_wrench)})")
    # Compared with an absolute tolerance: these are unix timestamps, so a relative
    # tolerance would accept differences of hours.
    if not (np.allclose(t_spatula, t_vial, rtol=0, atol=CLOCK_TOL_S)
            and np.allclose(t_spatula, t_wrench, rtol=0, atol=CLOCK_TOL_S)):
        apart = max(np.abs(t_spatula - t_vial).max(), np.abs(t_spatula - t_wrench).max())
        raise ValueError(f"{name}: the streams are not on a common clock "
                         f"(they differ by up to {apart:.3g} s)")

    vial_pos = np.array([_xyz(r["position"]) for r in vial])
    vial_quat = np.array([_xyzw(r["orientation"]) for r in vial])
    if vial_pos.std(axis=0).max() > static_tol_m:
        raise ValueError(f"{name}: the vial moved during the recording "
                         f"(position spread {vial_pos.std(axis=0)} m)")
    middle = len(vial_quat) // 2
    turned = float(np.degrees(np.linalg.norm(
        (R.from_quat(vial_quat[middle]).inv() * R.from_quat(vial_quat)).as_rotvec(),
        axis=1).max()))
    if turned > static_tol_deg:
        raise ValueError(f"{name}: the vial turned by {turned:.2f} degrees during the "
                         f"recording, more than the {static_tol_deg} degree tolerance")

    contact = meta["contact_pose"]
    return Demonstration(
        name=name,
        dataset=dataset,
        time=t_spatula,
        spatula_pos=np.array([_xyz(r["position"]) for r in spatula]),
        spatula_quat=np.array([_xyzw(r["orientation"]) for r in spatula]),
        wrench=np.array([[*_xyz(r["force"]), *_xyz(r["torque"])] for r in wrench]),
        vial_lip_pos=np.median(vial_pos, axis=0),
        vial_lip_quat=vial_quat[len(vial_quat) // 2],
        contact_time=float(meta["contact_time"]),
        contact_pos_mocap=_xyz(contact["position"]),
        contact_quat_mocap=_xyzw(contact["orientation"]),
        sampled_mass_mg=float(meta["scoop_weight_mg"]),
        descriptors=meta.get("descriptors", {}),
        rate_hz=float(meta.get("recording_rate_hz", 250.0)),
    )


def load_dataset(root: str) -> List[Demonstration]:
    """Read every demonstration under ``root``, sorted by folder name."""
    if not os.path.isdir(root):
        raise ValueError(f"dataset directory not found: {root}")
    demos = [load_demonstration(os.path.join(root, entry))
             for entry in sorted(os.listdir(root))
             if is_demonstration(os.path.join(root, entry))]
    if not demos:
        raise ValueError(f"no demonstration folders under {root}; each one must contain "
                         f"{', '.join(REQUIRED_FILES)}")
    return demos


def dataset_name(root: str) -> str:
    """The dataset's directory name, used to label outputs."""
    return os.path.basename(os.path.normpath(root))
