"""The task-parameterised model of a stroke.

Every demonstration is seen from each task frame at once: the state
``[position, orientation, velocity]`` is read in the vial frame and in the contact
frame, and the two views are stacked into one observation. Fitting a hidden Markov
model to those stacked observations gives, per state, one Gaussian per frame --
the description of what the stroke looks like from that frame at that phase of the
motion. The frames differ in the position and the linear velocity; the orientation
deviation is a body-frame quantity and is the same in both, so the product below
combines two views of where the spatula is and one of how it is turned.

What the frames buy is generalisation. A state whose vial-frame Gaussian is tight
is a part of the motion pinned to the vial; a state whose contact-frame Gaussian
is tight is pinned to where the material is touched. For a new vial pose and a new
contact point the frames are rebuilt, each frame's Gaussians are mapped into world
through its own transform, and their product weights every state by how certain
each frame is about it (:mod:`tplqt.synthesize`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from . import orientation as ori
from .frames import Frame, lift_frame
from .gaussian import GaussianMixture
from .hmm import HiddenMarkovModel
from .preprocess import Stroke

FRAMES: Tuple[str, ...] = ("vial", "contact")
ORIENTATION_FRAMES = ("vial", "contact", "world")


def chart_reference(strokes: Sequence[Stroke], orientation_frame: str) -> np.ndarray:
    """Reference orientation of the chart, for the chosen anchoring frame.

    Anchored in a task frame, the reference is the mean spatula orientation *read
    in that frame*, so it no longer depends on where the vial happened to stand
    during the recordings.
    """
    if orientation_frame == "world":
        return ori.reference_orientation([s.quat_world for s in strokes])
    relative = [(R.from_matrix(s.frames[orientation_frame][0]).inv()
                 * R.from_quat(s.quat_world)).as_quat() for s in strokes]
    return ori.reference_orientation(relative)


def configuration(stroke: Stroke, q_ref: Optional[np.ndarray], orientation_frame: str):
    """Configuration and its velocity for one stroke.

    Without orientation the configuration is the tip position. With orientation it
    is the position followed by the chart coordinates of the spatula orientation,
    read in the anchoring frame when the chart is anchored in one. The chart signal
    is smoothed before it is differentiated, as position is.
    """
    if q_ref is None:
        return stroke.pos_world, stroke.vel_world

    if orientation_frame == "world":
        quat = stroke.quat_world
    else:
        A_frame = stroke.frames[orientation_frame][0]
        quat = (R.from_matrix(A_frame).inv() * R.from_quat(stroke.quat_world)).as_quat()
    eta = ori.smooth_chart(ori.to_chart(quat, q_ref), stroke.rate_hz)
    eta_dot = ori.chart_velocity(eta, 1.0 / stroke.rate_hz)
    return (np.concatenate([stroke.pos_world, eta], axis=1),
            np.concatenate([stroke.vel_world, eta_dot], axis=1))


def lift_task_frame(frame: Frame, *, with_orientation: bool, orientation_frame: str,
                    n_deriv: int = 2) -> Frame:
    """Extend a task frame to the model's state.

    Position rotates and translates with the frame. The orientation block is the
    identity when the chart is anchored in a task frame, where the deviation is
    already expressed and must not be rotated again, and the frame's rotation when
    it is anchored in the world (see :mod:`tplqt.orientation`). Derivatives are
    lifted the same way as the configuration.
    """
    A, b = frame
    if not with_orientation:
        return lift_frame(A, b, n_deriv)
    orientation_block = A if orientation_frame == "world" else np.eye(3)
    return lift_frame(block_diag(A, orientation_block),
                      np.concatenate([np.asarray(b, float), np.zeros(3)]), n_deriv)


def observation(stroke: Stroke, *, q_ref: Optional[np.ndarray], orientation_frame: str,
                frames: Sequence[str] = FRAMES, n_deriv: int = 2):
    """Stack one stroke as it is seen from each task frame.

    Returns the ``(T, n_state * n_frames)`` observation and the lifted frames it
    was built with, which map each frame's coordinates back to world.
    """
    config, config_vel = configuration(stroke, q_ref, orientation_frame)
    state = np.concatenate([config, config_vel], axis=1)

    blocks, lifted = [], []
    for name in frames:
        A, b = lift_task_frame(stroke.frames[name], with_orientation=q_ref is not None,
                               orientation_frame=orientation_frame, n_deriv=n_deriv)
        blocks.append((state - b) @ A)              # row-wise A.T (state - b)
        lifted.append((A, b))
    return np.concatenate(blocks, axis=1), lifted


def state_schedule(hmm: HiddenMarkovModel, observations: Sequence[np.ndarray]):
    """Canonical order and dwell of the states, read off the training data.

    Each demonstration is segmented with the Viterbi algorithm. States are ordered
    by the average phase of the motion at which they are active, and each is given
    the average fraction of the stroke it occupies. This is the timing a new
    trajectory is generated with, so synthesis copies no single demonstration.
    """
    n_states = hmm.n_states
    phase_sum = np.zeros(n_states)
    visits = np.zeros(n_states)
    occupancy = np.zeros(n_states)
    for obs in observations:
        states = hmm.viterbi(obs)
        phase = np.linspace(0.0, 1.0, len(states))
        for k in range(n_states):
            active = states == k
            if active.any():
                phase_sum[k] += phase[active].mean()
                visits[k] += 1
                occupancy[k] += active.mean()

    mean_phase = np.where(visits > 0, phase_sum / np.maximum(visits, 1), np.inf)
    fraction = occupancy / len(observations)
    order = [int(k) for k in np.argsort(mean_phase) if visits[k] > 0]
    total = sum(fraction[k] for k in order)
    return [(k, float(fraction[k] / total)) for k in order]


def expand_schedule(schedule, horizon: int) -> np.ndarray:
    """Turn ``[(state, fraction), ...]`` into one state index per timestep.

    Every state gets at least one timestep, so the horizon must be at least as long
    as the schedule.
    """
    if horizon < len(schedule):
        raise ValueError(f"a horizon of {horizon} cannot hold {len(schedule)} states, "
                         "each of which needs at least one timestep")
    states = [s for s, _ in schedule]
    fractions = np.array([f for _, f in schedule], float)
    fractions = fractions / fractions.sum()
    counts = np.maximum(1, np.round(fractions * horizon).astype(int))
    while counts.sum() > horizon:
        counts[int(np.argmax(counts))] -= 1
    while counts.sum() < horizon:
        counts[int(np.argmax(fractions))] += 1
    return np.concatenate([np.full(c, s, int) for s, c in zip(states, counts)])[:horizon]


@dataclass
class TaskParameterizedModel:
    """A fitted model of the stroke, with everything synthesis needs.

    Attributes
    ----------
    hmm : HiddenMarkovModel
        The model over the stacked per-frame observations.
    frames : tuple of str
        Task frames, in the order they are stacked in an observation.
    q_ref : (4,) array or None
        Chart reference orientation, ``None`` for a position-only model.
    orientation_frame : str
        Frame the orientation chart is anchored in.
    contact_orientation : str
        Orientation convention of the contact frame the model was fitted with.
    schedule : list of (int, float)
        State order and dwell fractions.
    rate_hz, horizon : float, int
        Sampling rate of the demonstrations and their average length, the defaults
        used when generating a trajectory.
    chart_span_deg : float
        How far the demonstrations stray from the chart reference, in degrees. It
        is the scale on which a start orientation counts as far from anything
        demonstrated, and the check on the small-angle assumption.
    """

    hmm: HiddenMarkovModel
    frames: Tuple[str, ...]
    q_ref: Optional[np.ndarray]
    orientation_frame: str
    contact_orientation: str
    schedule: List[Tuple[int, float]]
    rate_hz: float
    horizon: int
    chart_span_deg: float = 0.0
    n_deriv: int = 2                # the state is a configuration and its velocity

    @property
    def has_orientation(self) -> bool:
        return self.q_ref is not None

    @property
    def n_config(self) -> int:
        """Configuration size: 3 for position only, 6 with orientation."""
        return 6 if self.has_orientation else 3

    @property
    def n_state(self) -> int:
        """State size per frame, configuration and its derivatives."""
        return self.n_config * self.n_deriv

    @property
    def n_states(self) -> int:
        return self.hmm.n_states

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    def observation(self, stroke: Stroke):
        """Stack a stroke the way the model was fitted.

        The stroke must carry the contact-frame convention the model was fitted
        with: read through the other one, its contact-frame Gaussians describe a
        different frame and every number that follows is wrong.
        """
        if stroke.contact_orientation != self.contact_orientation:
            raise ValueError(
                f"this model was fitted on {self.contact_orientation!r} contact frames but "
                f"the stroke was prepared with a {stroke.contact_orientation!r} one; "
                "prepare it the way the model was fitted")
        return observation(stroke, q_ref=self.q_ref,
                           orientation_frame=self.orientation_frame,
                           frames=self.frames, n_deriv=self.n_deriv)

    def anchored_reference(self, frames: Dict[str, Frame]) -> Optional[np.ndarray]:
        """Chart reference for a given set of task frames.

        The frames are the ones the trajectory is being generated for, so the
        orientation is anchored to *that* vial or contact rather than to the ones
        that were recorded.
        """
        if not self.has_orientation or self.orientation_frame == "world":
            return self.q_ref
        A = frames[self.orientation_frame][0]
        return ori.anchored_reference(self.q_ref, self.orientation_frame, R.from_matrix(A))

    def frame_gaussians(self, lifted: Sequence[Frame]) -> GaussianMixture:
        """Product over frames of the per-state Gaussians, mapped into world.

        Each frame's block of the observation is marginalised out, pushed into
        world through that frame's lifted transform, and the frames are multiplied
        state by state. The result is one Gaussian per state in world coordinates,
        tight where the frames agree. The frames carry the same orientation
        deviation, so what the product weighs is where each frame expects the
        spatula to be; the orientation follows the correlations each frame has
        learned between the two.
        """
        n_state = self.n_state
        product = None
        for index, (A, b) in enumerate(lifted):
            block = slice(index * n_state, (index + 1) * n_state)
            marginal = self.hmm.emissions.marginal(block).transform(A, b)
            product = marginal if product is None else product * marginal
        return product


def fit(strokes: Sequence[Stroke], *, n_states: int = 6, orientation: bool = True,
        orientation_frame: Optional[str] = None, frames: Sequence[str] = FRAMES,
        reg: float = 1e-6, max_iter: int = 100,
        verbose: bool = False) -> TaskParameterizedModel:
    """Fit the task-parameterised model to a set of prepared strokes.

    Parameters
    ----------
    strokes : list of Stroke
        Demonstrations prepared by :func:`tplqt.preprocess.prepare`; they must all
        use the same contact-frame convention.
    n_states : int
        Number of states of the hidden Markov model, the number of Gaussians per
        task frame.
    orientation : bool
        Model orientation alongside position.
    orientation_frame : str, optional
        Frame the orientation chart is anchored in, one of ``"vial"``,
        ``"contact"`` or ``"world"``. Defaults to ``"contact"`` when the contact
        frame is radial, so that the orientation follows the contact's azimuth,
        and to ``"vial"`` otherwise.
    reg : float
        Variance added to the covariances at each maximisation step.

    Returns
    -------
    TaskParameterizedModel
    """
    conventions = {s.contact_orientation for s in strokes}
    if len(conventions) != 1:
        raise ValueError(f"the strokes disagree on the contact-frame orientation "
                         f"({sorted(conventions)}); prepare them all the same way")
    contact_orientation = conventions.pop()

    if not orientation:
        chosen_frame = "world"
    elif orientation_frame is None:
        chosen_frame = "contact" if contact_orientation == "radial" else "vial"
    else:
        chosen_frame = orientation_frame

    if chosen_frame not in ORIENTATION_FRAMES:
        raise ValueError(f"orientation_frame must be one of {ORIENTATION_FRAMES}, "
                         f"got {chosen_frame!r}")
    if orientation and chosen_frame == "world" and contact_orientation == "radial":
        raise ValueError(
            "a radial contact frame cannot be combined with the world orientation chart: "
            "the world chart rotates the orientation block by each frame's rotation, so "
            "the contact frame's azimuth would leak into the orientation. Anchor the chart "
            "in the vial or the contact frame instead")
    if orientation and chosen_frame == "contact" and contact_orientation != "radial":
        raise ValueError(
            "the contact orientation chart needs a radial contact frame: anchored to a "
            "bore-aligned contact frame it is identical to the vial chart. Prepare the "
            "strokes with contact_orientation='radial'")

    q_ref = chart_reference(strokes, chosen_frame) if orientation else None
    observations = [observation(s, q_ref=q_ref, orientation_frame=chosen_frame,
                                frames=frames)[0] for s in strokes]
    chart_span = 0.0
    if orientation:
        chart_span = max(ori.chart_excursion_deg(
            configuration(s, q_ref, chosen_frame)[0][:, 3:6]) for s in strokes)
    hmm = HiddenMarkovModel.fit(observations, n_states, reg=reg, max_iter=max_iter,
                                verbose=verbose)

    return TaskParameterizedModel(
        hmm=hmm,
        frames=tuple(frames),
        q_ref=q_ref,
        orientation_frame=chosen_frame,
        contact_orientation=contact_orientation,
        schedule=state_schedule(hmm, observations),
        rate_hz=float(strokes[0].rate_hz),
        horizon=int(round(np.mean([s.n_samples for s in strokes]))),
        chart_span_deg=float(chart_span),
    )
