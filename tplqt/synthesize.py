"""Generating a stroke for a situation that was never demonstrated.

Given a vial pose and a contact point, the task frames are rebuilt, each frame's
Gaussians are mapped into world through its own transform and multiplied state by
state, and the resulting sequence of Gaussians is tracked by the linear quadratic
tracker. The timing comes from the model's own state schedule, so no single
demonstration is replayed.

Moving the vial rigidly moves the whole stroke with it, because both frames are
attached to the vial. What makes the model adapt rather than translate is the
contact point: it moves the contact frame relative to the vial frame, so the
product of Gaussians has to trade the two off and the stroke is re-planned.

The generated stroke can be required to keep the spatula inside the vial; see
:mod:`tplqt.safety`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import orientation as ori
from .frames import Frame, contact_frame, project_position, quat_from_tilt
from .lqt import canonical_system, solve_lqt
from .model import FRAMES, TaskParameterizedModel, expand_schedule, lift_task_frame
from .preprocess import Stroke
from .safety import SafetySettings, constrain_to_vial, wall_radius

# Variance the pinned coordinates of the reference are shrunk to, in square metres
# for position and in square radians for orientation.
PINNED_POSITION_VARIANCE = 1e-8
PINNED_ORIENTATION_VARIANCE = 1e-8

# A start orientation further from the one the model expects than this multiple of the
# span the demonstrations cover is an extrapolation, and so is one that drags the start
# of the stroke further than this.
START_SPAN_FACTOR = 1.5
START_DRAG_M = 0.05


@dataclass
class Segment:
    """One run of timesteps spent in the same state."""

    state: int
    start: int
    end: int
    rms: float

    @property
    def n_samples(self) -> int:
        return self.end - self.start + 1


@dataclass
class Synthesis:
    """A generated stroke."""

    xi: np.ndarray                          # (T, 2 * n_config) state in world
    states: np.ndarray                      # (T,) state tracked at each timestep
    pos_world: np.ndarray                   # (T, 3) tip position in world
    pos_vial: np.ndarray                    # (T, 3) tip position in the vial frame
    vel_world: np.ndarray                   # (T, 3) tip velocity in world
    frames: Dict[str, Frame]                # the task frames it was generated for
    jerk_rms: float                         # smoothness, m/s^3
    tracking_rms: float                     # distance to the tracked Gaussian means, m
    segments: List[Segment]                 # tracking error per state segment
    dt: float                               # timestep between samples
    quat_world: Optional[np.ndarray] = None  # (T, 4) orientation, when modelled
    omega_world: Optional[np.ndarray] = None  # (T, 3) angular velocity in world
    eta: Optional[np.ndarray] = None         # (T, 3) chart coordinates, when modelled
    constrained: bool = False               # whether the geometric constraints were solved
    solver_status: str = ""                 # status reported by the constrained solve

    @property
    def time(self) -> np.ndarray:
        """(T,) timestamps of the stroke, starting at zero."""
        return np.arange(len(self.xi)) * self.dt


def _start_drag(x0, mu, sigma) -> float:
    """How far the reference's position moves once the start orientation is known.

    Position and orientation are correlated within a state, so conditioning the
    first reference Gaussian on the start orientation shifts its position mean by
    ``Sigma_pe Sigma_ee^-1 (eta_0 - mu_eta)``. That shift is what drags the stroke
    when the start orientation is one the model never saw.
    """
    deviation = x0[3:6] - mu[3:6]
    return float(np.linalg.norm(
        sigma[0:3, 3:6] @ np.linalg.solve(sigma[3:6, 3:6], deviation)))


def angular_velocity(eta, eta_dot, quat_world) -> np.ndarray:
    """World angular velocity from the chart state.

    The chart rate is not itself an angular velocity; the right Jacobian of
    :mod:`tplqt.orientation` maps it to the body rate, ``omega_body = J_r(eta)
    eta_dot``, which is then rotated into world. Taking it from the tracker's state
    rather than differencing the generated orientations keeps it smooth and
    consistent with the dynamics.
    """
    body = np.einsum("tij,tj->ti", ori.right_jacobian(eta), eta_dot)
    return np.einsum("tij,tj->ti", R.from_quat(quat_world).as_matrix(), body)


def task_frames(vial_pose, contact_point, *, frames: Sequence[str] = FRAMES,
                contact_orientation: str = "vial") -> Dict[str, Frame]:
    """Build the task frames of a situation.

    ``vial_pose`` is ``(position, quaternion)`` in world and ``contact_point`` is
    given in the vial frame, measured from the centre of the lip. The contact
    frame's orientation convention must be the one the model was fitted with.
    """
    position, quaternion = vial_pose
    A = R.from_quat(quaternion).as_matrix()
    b = np.asarray(position, float)
    contact_world = A @ np.asarray(contact_point, float) + b
    built = {
        "vial": (A, b),
        "contact": contact_frame(A, contact_world, contact_point,
                                 orientation=contact_orientation),
    }
    return {name: built[name] for name in frames}


def _segments(pos_world, reference_pos, states) -> List[Segment]:
    """Tracking error broken down by run of constant state."""
    error = np.linalg.norm(pos_world - reference_pos, axis=1)
    out, start = [], 0
    while start < len(states):
        end = start
        while end < len(states) and states[end] == states[start]:
            end += 1
        out.append(Segment(state=int(states[start]), start=int(start), end=int(end - 1),
                           rms=float(np.sqrt((error[start:end] ** 2).mean()))))
        start = end
    return out


def _pin_block(mu_seq, sigma_seq, timesteps, block: slice, value, variance: float) -> None:
    """Pin part of the reference at the given timesteps, tightly.

    The pinned coordinates are decoupled from the rest of the state before their
    variance is shrunk, which keeps the covariance positive definite.
    """
    for t in timesteps:
        mu_seq[t, block] = value
        sigma_seq[t, block, :] = 0.0
        sigma_seq[t, :, block] = 0.0
        width = block.stop - block.start
        sigma_seq[t, block, block] = np.eye(width) * variance


def synthesize(model: TaskParameterizedModel, vial_pose, contact_point, *,
               horizon: Optional[int] = None, dt: Optional[float] = None,
               control_cost: float = 1.0, schedule=None,
               contact_orientation: Optional[str] = None,
               start: Optional[np.ndarray] = None,
               start_shift=(0.0, 0.0, 0.0), tilt: Optional[float] = None,
               boundary_orientation=None, anchor_end: bool = False,
               anchor_steps: int = 12,
               safety: Optional[SafetySettings] = None) -> Synthesis:
    """Generate a stroke for a vial pose and a contact point.

    Parameters
    ----------
    model : TaskParameterizedModel
        A fitted model.
    vial_pose : (position, quaternion)
        The vial's pose in world, quaternion in xyzw order.
    contact_point : (3,) array
        Where the spatula should meet the material, in the vial frame, measured
        from the centre of the lip.
    horizon, dt : int, float
        Number of timesteps and timestep; both default to the model's.
    control_cost : float
        Weight on control effort. Raising it smooths the stroke at the cost of
        tracking, which on a sampling task means missing the contact point.
    schedule : list of (int, float), optional
        State order and dwell fractions; the model's schedule by default.
    contact_orientation : str, optional
        Contact-frame convention. It must be the one the model was fitted with,
        which is what it defaults to; naming a different one is an error rather
        than a silent change of frame.
    start : (2 * n_config,) array, optional
        Initial state. By default the stroke starts at rest at the centre of the
        lip, displaced by ``start_shift``, holding the orientation the model's
        first state expects.
    start_shift : (3,) array
        Where the stroke starts, relative to the centre of the lip, in the vial
        frame.
    tilt : float, optional
        Start the spatula at this angle from gravity instead, pointing into the
        vial; 90 degrees is horizontal. Two things follow from asking for a tilt.
        It is measured from gravity, a world direction, so a stroke started that way
        follows the vial only up to rotations about the vertical, while the model
        itself is equivariant. And a tilt far from the demonstrated one asks the
        model to extrapolate: position and orientation are correlated within a
        state, so an unexpected start orientation pulls the whole stroke with it,
        which is reported as a warning.
    boundary_orientation : (4,) array, optional
        Orientation to hold at both ends of the stroke, as a world quaternion. It
        pins the start and the last ``anchor_steps`` timesteps, so that entry and
        exit are at a fixed pose and the model shapes only the middle of the
        stroke.
    anchor_end : bool
        Require the stroke to end at the centre of the lip, so that the spatula
        leaves the vial.
    anchor_steps : int
        How many timesteps each anchor holds: the last ``anchor_steps`` of the
        stroke for ``anchor_end``, and the first and last ``anchor_steps`` for
        ``boundary_orientation``. Both are clipped to the horizon.
    safety : SafetySettings, optional
        When given, the stroke is re-solved under the constraint that the spatula
        stays inside the vial and passes through its opening
        (:func:`tplqt.safety.constrain_to_vial`).

    Returns
    -------
    Synthesis
    """
    if (contact_orientation is not None
            and contact_orientation != model.contact_orientation):
        raise ValueError(
            f"this model was fitted with a {model.contact_orientation!r} contact frame; "
            f"generating with a {contact_orientation!r} one would read its contact-frame "
            "Gaussians in the wrong frame. Prepare the strokes with the convention you "
            "want and fit again")
    frames = task_frames(vial_pose, contact_point, frames=model.frames,
                         contact_orientation=model.contact_orientation)
    vial_position = np.asarray(vial_pose[0], float)
    vial_quaternion = np.asarray(vial_pose[1], float)
    A_vial, b_vial = frames["vial"]

    horizon = int(horizon) if horizon is not None else model.horizon
    if horizon < max(2, len(model.schedule)):
        raise ValueError(f"a horizon of {horizon} is too short: the stroke needs at least "
                         f"one timestep per state, and this model has {len(model.schedule)}")
    dt = float(dt) if dt is not None else model.dt
    schedule = schedule if schedule is not None else model.schedule
    states = expand_schedule(schedule, horizon)

    lifted = [lift_task_frame(frames[name], with_orientation=model.has_orientation,
                              orientation_frame=model.orientation_frame,
                              n_deriv=model.n_deriv) for name in model.frames]
    product = model.frame_gaussians(lifted)
    mu_seq, sigma_seq = product.sequence(states)
    mu_seq, sigma_seq = mu_seq.copy(), sigma_seq.copy()

    q_ref = model.anchored_reference(frames)

    end_steps = max(1, min(int(anchor_steps), horizon - 1)) if anchor_end else 0
    if anchor_end:
        _pin_block(mu_seq, sigma_seq, range(horizon - end_steps, horizon),
                   slice(0, 3), vial_position, PINNED_POSITION_VARIANCE)

    boundary_eta = None
    if boundary_orientation is not None and model.has_orientation:
        boundary_eta = np.atleast_2d(ori.to_chart(
            np.asarray(boundary_orientation, float), q_ref))[0]
        steps = max(1, min(int(anchor_steps), horizon // 2))
        _pin_block(mu_seq, sigma_seq,
                   list(range(steps)) + list(range(horizon - steps, horizon)),
                   slice(3, 6), boundary_eta, PINNED_ORIENTATION_VARIANCE)

    if not model.has_orientation and (tilt is not None or boundary_orientation is not None):
        raise ValueError("a tilt or a boundary orientation needs a model fitted with "
                         "orientation; this one models position only")

    if start is not None:
        x0 = np.asarray(start, float)
    else:
        x0 = np.zeros(2 * model.n_config)
        x0[:3] = vial_position + A_vial @ np.asarray(start_shift, float)
        if model.has_orientation:
            if boundary_eta is not None:
                x0[3:6] = boundary_eta
            elif tilt is None:
                x0[3:6] = product.mu[states[0]][3:6]
            else:
                expected = ori.from_chart(product.mu[states[0]][3:6], q_ref)
                x0[3:6] = ori.to_chart(
                    quat_from_tilt(vial_quaternion, float(tilt), expected), q_ref)

    if model.has_orientation:
        # A start orientation the demonstrations never reached is an extrapolation,
        # and because position and orientation are correlated within a state it also
        # drags the start of the stroke away from where the model would put it.
        away_deg = float(np.degrees(np.linalg.norm(x0[3:6] - product.mu[states[0]][3:6])))
        drag = _start_drag(x0, mu_seq[0], sigma_seq[0])
        if (away_deg > START_SPAN_FACTOR * model.chart_span_deg or drag > START_DRAG_M):
            warnings.warn(
                f"the start orientation is {away_deg:.0f} degrees from the one the model "
                f"expects here, against the {model.chart_span_deg:.0f} degrees the "
                f"demonstrations span, and it pulls the start of the stroke by about "
                f"{drag * 1e3:.0f} mm; start from the model's own orientation (tilt=None) "
                "or from a pose closer to it", stacklevel=2)

    A_dyn, B_dyn = canonical_system(model.n_config, model.n_deriv, dt)
    xi = solve_lqt(A_dyn, B_dyn, x0, mu_seq, sigma_seq, control_cost)

    solver_status = ""
    if safety is not None:
        if not model.has_orientation:
            raise ValueError("the geometric constraints need a model fitted with "
                             "orientation, since they act on the spatula's axis")
        contact = np.asarray(contact_point, float)
        reach = float(np.hypot(contact[0], contact[1]))
        allowed = float(wall_radius(contact[2], safety.vial)
                        - safety.spatula.radius - safety.margin)
        if reach > allowed:
            warnings.warn(
                f"the contact point is {reach * 1e3:.1f} mm from the bore axis but the "
                f"constraints put the wall at {allowed * 1e3:.1f} mm at that depth, so the "
                "stroke will stop at the wall instead of reaching it", stacklevel=2)
        xi, solver_status = constrain_to_vial(
            xi, A_dyn=A_dyn, B_dyn=B_dyn, x0=x0, mu_seq=mu_seq, sigma_seq=sigma_seq,
            control_cost=control_cost, vial_frame=(A_vial, b_vial), q_ref=q_ref,
            settings=safety,
            anchor_end_steps=end_steps,
            anchor_target=vial_position)

    pos_world = xi[:, :3]
    vel_world = xi[:, model.n_config:model.n_config + 3]
    quat_world = omega_world = eta = None
    if model.has_orientation:
        eta = xi[:, 3:6]
        quat_world = ori.from_chart(eta, q_ref)
        omega_world = angular_velocity(eta, xi[:, model.n_config + 3:model.n_config + 6],
                                       quat_world)

    jerk = np.gradient(np.gradient(np.gradient(pos_world, dt, axis=0), dt, axis=0),
                       dt, axis=0)
    reference_pos = mu_seq[:, :3]
    return Synthesis(
        xi=xi,
        states=states,
        pos_world=pos_world,
        pos_vial=project_position(pos_world, A_vial, b_vial),
        vel_world=vel_world,
        frames=frames,
        jerk_rms=float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))),
        tracking_rms=float(np.sqrt(np.mean(np.sum((pos_world - reference_pos) ** 2, axis=1)))),
        segments=_segments(pos_world, reference_pos, states),
        dt=dt,
        quat_world=quat_world,
        omega_world=omega_world,
        eta=eta,
        constrained=safety is not None,
        solver_status=solver_status,
    )


def contact_cluster(strokes: Sequence[Stroke]) -> np.ndarray:
    """The recorded contact points in the vial frame, ``(N, 3)``."""
    return np.array([s.contact_pos_vial for s in strokes])


def mean_contact(strokes: Sequence[Stroke]) -> np.ndarray:
    """Centre of the recorded contact points in the vial frame."""
    return contact_cluster(strokes).mean(axis=0)


def path_rms_by_state(model: TaskParameterizedModel, strokes: Sequence[Stroke],
                      synthesis: Synthesis):
    """How close a generated stroke stays to the demonstrations, state by state.

    For each timestep, the distance to the nearest demonstrated point *assigned to
    the same state* is measured in the vial frame. Distance to a state's mean would
    mostly measure how much ground that state covers; distance to the demonstrated
    points measures whether the generated stroke stays on the demonstrated
    manifold. The spread of the demonstrations within the state is reported
    alongside as the scale to read it against.

    Returns ``(rows, overall)`` where each row is
    ``{state, n_generated, n_demonstrated, rms, demonstration_spread}`` in metres.
    """
    demonstrated: Dict[int, List[np.ndarray]] = {}
    for stroke in strokes:
        obs, _ = model.observation(stroke)
        states = model.hmm.viterbi(obs)
        for state in np.unique(states):
            demonstrated.setdefault(int(state), []).append(stroke.pos_vial[states == state])
    points = {state: np.concatenate(parts) for state, parts in demonstrated.items()}

    rows, pooled = [], []
    for segment in synthesis.segments:
        generated = synthesis.pos_vial[segment.start:segment.end + 1]
        reference = points.get(segment.state)
        if reference is None or len(reference) == 0:
            rows.append({"state": segment.state, "n_generated": segment.n_samples,
                         "n_demonstrated": 0, "rms": float("nan"),
                         "demonstration_spread": float("nan")})
            continue
        distance = np.linalg.norm(generated[:, None, :] - reference[None, :, :],
                                  axis=2).min(axis=1)
        pooled.extend(distance.tolist())
        centred = reference - reference.mean(axis=0)
        rows.append({
            "state": segment.state,
            "n_generated": segment.n_samples,
            "n_demonstrated": len(reference),
            "rms": float(np.sqrt((distance ** 2).mean())),
            "demonstration_spread": float(np.sqrt(np.mean(np.sum(centred ** 2, axis=1)))),
        })
    overall = float(np.sqrt(np.mean(np.array(pooled) ** 2))) if pooled else float("nan")
    return rows, overall
