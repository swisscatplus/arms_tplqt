"""Replaying a demonstration with the fitted model.

Reproduction is the fidelity check: the model is given one of the demonstrations
it was fitted on -- its task frames and its own state segmentation -- and asked to
regenerate the stroke. The gap between the demonstration and its reproduction is
what the model lost by compressing the demonstrations into a handful of Gaussians
per frame. Generating a stroke for a situation that was never demonstrated is
:mod:`tplqt.synthesize`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import orientation as ori
from .frames import project_position
from .lqt import canonical_system, solve_lqt
from .model import TaskParameterizedModel, configuration
from .preprocess import Stroke


@dataclass
class Reproduction:
    """A demonstration as the model regenerates it."""

    xi: np.ndarray                          # (T, 2 * n_config) state in world
    states: np.ndarray                      # (T,) state active at each timestep
    pos_world: np.ndarray                   # (T, 3) tip position in world
    pos_vial: np.ndarray                    # (T, 3) tip position in the vial frame
    contact_state: int                      # state active at the contact instant
    quat_world: Optional[np.ndarray] = None  # (T, 4) orientation, when modelled
    eta: Optional[np.ndarray] = None         # (T, 3) chart coordinates, when modelled


def reproduce(model: TaskParameterizedModel, stroke: Stroke, *,
              control_cost: float = 1e-4, dt: Optional[float] = None) -> Reproduction:
    """Regenerate ``stroke`` with ``model``.

    The state sequence is the one the model itself assigns to the demonstration,
    so this measures how well the Gaussians and the tracker represent a stroke,
    not how well the timing is predicted. ``control_cost`` is low by default, which
    tracks the demonstration closely.
    """
    observation, lifted = model.observation(stroke)
    product = model.frame_gaussians(lifted)
    states = model.hmm.viterbi(observation)
    mu_seq, sigma_seq = product.sequence(states)

    dt = float(dt) if dt is not None else 1.0 / stroke.rate_hz
    A, B = canonical_system(model.n_config, model.n_deriv, dt)
    config, config_vel = configuration(stroke, model.q_ref, model.orientation_frame)
    x0 = np.concatenate([config[0], config_vel[0]])
    xi = solve_lqt(A, B, x0, mu_seq, sigma_seq, control_cost)

    A_vial, b_vial = stroke.frames["vial"]
    quat_world = eta = None
    if model.has_orientation:
        eta = xi[:, 3:6]
        quat_world = ori.from_chart(eta, model.anchored_reference(stroke.frames))

    return Reproduction(
        xi=xi,
        states=states,
        pos_world=xi[:, :3],
        pos_vial=project_position(xi[:, :3], A_vial, b_vial),
        contact_state=int(states[stroke.contact_index]),
        quat_world=quat_world,
        eta=eta,
    )


def position_rms(stroke: Stroke, reproduction: Reproduction) -> float:
    """Position error between a demonstration and its reproduction, in metres.

    Measured in the vial frame, which is where the error matters: it is the offset
    from the vial the spatula would have had.
    """
    error = reproduction.pos_vial - stroke.pos_vial
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


def orientation_rms_deg(stroke: Stroke, reproduction: Reproduction) -> float:
    """Orientation error between a demonstration and its reproduction, in degrees."""
    if reproduction.quat_world is None:
        raise ValueError("this reproduction has no orientation; fit with orientation=True")
    angle = ori.geodesic_angle_deg(stroke.quat_world, reproduction.quat_world)
    return float(np.sqrt(np.mean(angle ** 2)))
