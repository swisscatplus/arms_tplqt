"""Turning a recording into the stroke the model is fitted on.

Each demonstration is recorded from before the spatula enters the vial until
after it leaves, at the motion-capture rate. Four steps prepare it:

1. window the recording to the segment the tip spends inside the vial, the part
   the task frames describe;
2. smooth the tip position at the recorded rate, which also acts as the
   anti-aliasing filter for step 3;
3. downsample to the control rate;
4. differentiate the smoothed, downsampled position to get velocity.

Differentiating before smoothing would amplify the motion-capture noise, so the
order matters. Orientation is carried through the same windowing and downsampling
and is smoothed in :mod:`tplqt.orientation`, where it has a vector representation;
that smoothing happens after the downsampling rather than before it, so it removes
noise but cannot undo aliasing.

The rate used for the derivative is the rate the recording was configured at,
divided by the decimation stride. The motion-capture timestamps wander a few per
cent around that nominal rate, so velocities carry the same few per cent; the
alternative, differentiating on the measured timestamps, would make the velocity
scale differ from demonstration to demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .calibration import calibration_for_dataset
from .dataset import Demonstration
from .frames import (Frame, contact_frame, ensure_quaternion_continuity, frame_from_pose,
                     mocap_to_world, project_position, vial_bore_axis)


@dataclass
class Stroke:
    """One demonstration reduced to its in-vial segment, with its task frames.

    Positions and velocities are in world; ``frames`` holds the task frames of
    this demonstration, and ``pos_vial`` is the tip path read in the vial frame,
    which is the frame the reproduction error is reported in.
    """

    name: str
    rate_hz: float

    time: np.ndarray                # (T,) seconds
    pos_world: np.ndarray           # (T, 3) smoothed, downsampled tip position
    quat_world: np.ndarray          # (T, 4) xyzw, sign-continuous
    vel_world: np.ndarray           # (T, 3) tip velocity
    wrench: np.ndarray              # (T, 6) force and torque

    frames: Dict[str, Frame]        # {"vial": (A, b), "contact": (A, b)} in world
    vial_lip_pos: np.ndarray        # (3,) lip centre in world
    bore_axis: np.ndarray           # (3,) bore direction in world
    contact_index: int              # index of the contact instant within the stroke
    contact_pos_vial: np.ndarray    # (3,) contact point in the vial frame
    pos_vial: np.ndarray            # (T, 3) tip path in the vial frame
    contact_orientation: str        # orientation convention of the contact frame

    sampled_mass_mg: float = 0.0
    descriptors: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.time.shape[0])


def in_vial_window(tip_pos_world, time, bore_axis, lip_pos_world, contact_time):
    """First and last index (inclusive) of the stroke inside the vial.

    Depth along the bore is ``(p - lip) . bore``, negative inside the vial. The
    stroke is the run of samples with negative depth that contains the contact
    instant, so a tip that dips in and out several times contributes only the
    visit in which the material was touched.
    """
    depth = (np.asarray(tip_pos_world) - lip_pos_world) @ bore_axis
    inside = depth < 0
    contact = int(np.argmin(np.abs(time - contact_time)))
    if not inside[contact]:
        raise ValueError(
            "the contact sample lies outside the vial opening, which means the recording "
            "and the vial pose are not in the same frame; check that the calibration "
            "matches the dataset")
    start = contact
    while start > 0 and inside[start - 1]:
        start -= 1
    end = contact
    while end < len(inside) - 1 and inside[end + 1]:
        end += 1
    return start, end


def _odd_window(n_samples: int, limit: int) -> int:
    """Largest odd window of at most ``n_samples`` that fits in ``limit`` samples."""
    window = min(int(n_samples) | 1, (limit - 1) | 1 if limit % 2 == 0 else limit)
    return max(window, 3)


def smooth(signal, rate_hz: float, window_s: float = 0.08, polyorder: int = 3) -> np.ndarray:
    """Zero-phase Savitzky-Golay smoothing of an ``(N, d)`` signal.

    Zero phase matters because the smoothed signal is differentiated: a filter
    with phase lag would shift the velocity in time. Signals too short for the
    window are returned unchanged.
    """
    signal = np.asarray(signal, float)
    window = _odd_window(round(window_s * rate_hz), signal.shape[0])
    if signal.shape[0] <= polyorder + 2 or window <= polyorder:
        return signal.copy()
    return savgol_filter(signal, window_length=window, polyorder=polyorder, axis=0)


def decimation_factor(rate_hz: float, target_rate: float) -> int:
    """Integer stride that brings ``rate_hz`` closest to ``target_rate``."""
    return max(1, int(round(rate_hz / float(target_rate))))


def prepare(demo: Demonstration, *, calibration: Optional[Tuple] = None,
            target_rate: float = 50.0, window_s: float = 0.08, polyorder: int = 3,
            contact_orientation: str = "vial") -> Stroke:
    """Prepare one demonstration for modelling.

    ``calibration`` is the ``(translation, quaternion)`` pair that lifts the
    motion-capture poses into world; it defaults to the calibration registered for
    the dataset the demonstration came from. The vial pose is measured in world
    already, so it does not depend on this calibration.

    ``contact_orientation`` selects the contact frame's orientation, ``"vial"`` or
    ``"radial"`` (see :func:`tplqt.frames.contact_frame`); the choice is recorded
    on the stroke so that fitting can check every demonstration agrees.
    """
    calib = calibration if calibration is not None else calibration_for_dataset(demo.dataset)

    pos_world, quat_world = mocap_to_world(demo.spatula_pos, demo.spatula_quat,
                                           calibration=calib)
    quat_world = ensure_quaternion_continuity(quat_world)
    lip = demo.vial_lip_pos
    bore = vial_bore_axis(demo.vial_lip_quat)

    start, end = in_vial_window(pos_world, demo.time, bore, lip, demo.contact_time)
    window = slice(start, end + 1)
    smoothed = smooth(pos_world[window], demo.rate_hz, window_s, polyorder)

    stride = decimation_factor(demo.rate_hz, target_rate)
    keep = np.arange(0, smoothed.shape[0], stride)
    time = demo.time[window][keep]
    pos = smoothed[keep]
    quat = ensure_quaternion_continuity(quat_world[window][keep])
    wrench = demo.wrench[window][keep]
    rate_hz = demo.rate_hz / stride
    vel = np.gradient(pos, 1.0 / rate_hz, axis=0)

    A_vial, b_vial = frame_from_pose(lip, demo.vial_lip_quat)
    contact_world = mocap_to_world(demo.contact_pos_mocap, calibration=calib)
    contact_pos_vial = project_position(contact_world, A_vial, b_vial)[0]
    frames = {
        "vial": (A_vial, b_vial),
        "contact": contact_frame(A_vial, contact_world, contact_pos_vial,
                                 orientation=contact_orientation),
    }

    return Stroke(
        name=demo.name,
        rate_hz=rate_hz,
        time=time,
        pos_world=pos,
        quat_world=quat,
        vel_world=vel,
        wrench=wrench,
        frames=frames,
        vial_lip_pos=lip,
        bore_axis=bore,
        contact_index=int(np.argmin(np.abs(time - demo.contact_time))),
        contact_pos_vial=contact_pos_vial,
        pos_vial=project_position(pos, A_vial, b_vial),
        contact_orientation=contact_orientation,
        sampled_mass_mg=demo.sampled_mass_mg,
        descriptors=dict(demo.descriptors),
    )


def prepare_dataset(demos, **kwargs):
    """Prepare every demonstration of a dataset with the same settings."""
    return [prepare(demo, **kwargs) for demo in demos]
