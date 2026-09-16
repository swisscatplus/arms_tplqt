"""Orientation as a vector, so that one linear model covers position and orientation.

Orientations live on SO(3), which the Gaussian model and the linear quadratic
tracker cannot represent directly. Both are given a vector to work with by
choosing a reference orientation ``q_ref`` and coordinatising every orientation by
its deviation from it,

    eta = Log(q_ref^-1 * q),       q = q_ref * Exp(eta),

a rotation vector in R^3. Inside this chart ``eta`` is treated as three further
configuration coordinates alongside position, which is exact for small deviations
and is what makes the whole Euclidean pipeline apply unchanged;
:func:`chart_excursion_deg` measures how far a dataset actually strays from the
reference, which is the check on that assumption.

The chart is anchored in a task frame rather than in the world: the reference is
stored relative to the vial or the contact frame and rotated back with the frame
at reconstruction time (:func:`anchored_reference`). The model is then
equivariant -- move the vial and the whole stroke, orientation included, moves with
it. Anchoring it in the world instead is a basis change with no such property, and
is kept as the baseline the frame-anchored chart is compared against.

The deviation is a body-frame quantity: ``Log(q_ref^-1 q)`` is unchanged when the
orientation and its reference are both read in the same frame. That is why a task
frame lifts to ``blockdiag(A, I)`` for the joint configuration when the chart is
anchored in a task frame -- position turns with the frame, the deviation is already
expressed there -- and it is what the equivariance of the model rests on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .preprocess import smooth


def reference_orientation(quats) -> np.ndarray:
    """Chart reference: the mean orientation of one or several demonstrations.

    ``quats`` is an ``(N, 4)`` array or a list of them; the mean is taken over all
    of them so that one chart is shared by the whole fit.
    """
    if isinstance(quats, (list, tuple)):
        stacked = np.concatenate([np.atleast_2d(np.asarray(q, float)) for q in quats], axis=0)
    else:
        stacked = np.atleast_2d(np.asarray(quats, float))
    return R.from_quat(stacked).mean().as_quat()


def to_chart(quat, reference) -> np.ndarray:
    """Chart coordinates ``Log(reference^-1 * quat)`` of one or many orientations."""
    return (R.from_quat(reference).inv() * R.from_quat(quat)).as_rotvec()


def from_chart(eta, reference) -> np.ndarray:
    """Orientation ``reference * Exp(eta)``, the inverse of :func:`to_chart`."""
    return (R.from_quat(reference) * R.from_rotvec(eta)).as_quat()


def anchored_reference(q_ref, orientation_frame: str, frame_rotation: Optional[R]):
    """The chart reference to reconstruct with, for a given task-frame rotation.

    With the chart anchored in the world frame the stored reference is used as it
    is. With it anchored in the vial or contact frame the stored reference is
    frame-relative, so the frame's rotation is applied to it:
    ``q_ref_eff = A_frame * q_ref``. ``frame_rotation`` is the rotation of the
    anchoring frame, as a :class:`~scipy.spatial.transform.Rotation`.
    """
    if q_ref is None or orientation_frame == "world":
        return q_ref
    if frame_rotation is None:
        raise ValueError(f"the {orientation_frame!r} chart needs the frame rotation "
                         "to anchor the reference orientation")
    return (frame_rotation * R.from_quat(q_ref)).as_quat()


def smooth_chart(eta, rate_hz: float, window_s: float = 0.08,
                 polyorder: int = 3) -> np.ndarray:
    """Smooth the chart signal, the orientation counterpart of position smoothing."""
    return smooth(np.asarray(eta, float), rate_hz, window_s, polyorder)


def chart_velocity(eta, dt: float) -> np.ndarray:
    """Rate of change of the chart coordinates.

    This is the derivative of the chart coordinate, not the body angular velocity;
    near the reference the two agree to first order, and
    :func:`right_jacobian` converts between them exactly.
    """
    return np.gradient(np.asarray(eta, float), dt, axis=0)


def geodesic_angle_deg(quat_a, quat_b) -> np.ndarray:
    """Angle in degrees between two orientations, elementwise over a batch."""
    relative = R.from_quat(quat_a).inv() * R.from_quat(quat_b)
    return np.degrees(np.linalg.norm(np.atleast_2d(relative.as_rotvec()), axis=1))


def chart_excursion_deg(eta) -> float:
    """Largest chart deviation in degrees, the check on the small-angle assumption."""
    return float(np.degrees(np.linalg.norm(np.atleast_2d(eta), axis=1).max()))


def _skew(v) -> np.ndarray:
    """Skew-symmetric matrix (or batch) of a 3-vector, so that ``skew(v) w = v x w``."""
    v = np.asarray(v, float)
    zero = np.zeros_like(v[..., 0])
    return np.stack([
        np.stack([zero,     -v[..., 2],  v[..., 1]], axis=-1),
        np.stack([v[..., 2],  zero,     -v[..., 0]], axis=-1),
        np.stack([-v[..., 1], v[..., 0], zero], axis=-1),
    ], axis=-2)


def right_jacobian(eta) -> np.ndarray:
    """Right Jacobian of ``Exp`` at ``eta``.

    It satisfies ``Exp(eta + d) = Exp(eta) Exp(J_r(eta) d)`` to first order, and

        J_r(eta) = I - (1 - cos t)/t^2 [eta]_x + (t - sin t)/t^3 [eta]_x^2,  t = |eta|,

    with the Taylor expansion used below ``1e-6`` to avoid the removable
    singularity at the identity. Accepts ``(3,)`` or ``(T, 3)``.
    """
    eta = np.asarray(eta, float)
    single = eta.ndim == 1
    eta = eta.reshape(-1, 3)
    angle = np.linalg.norm(eta, axis=-1)
    skew = _skew(eta)
    skew_sq = np.einsum("tij,tjk->tik", skew, skew)

    small = angle < 1e-6
    safe = np.where(small, 1.0, angle)
    a = np.where(small, 0.5 - angle ** 2 / 24.0, (1 - np.cos(angle)) / safe ** 2)
    b = np.where(small, 1.0 / 6 - angle ** 2 / 120.0, (angle - np.sin(angle)) / safe ** 3)
    jacobian = np.eye(3)[None] - a[:, None, None] * skew + b[:, None, None] * skew_sq
    return jacobian[0] if single else jacobian


def tool_axis_jacobian(eta, reference) -> np.ndarray:
    """Derivative of the world tool axis with respect to the chart coordinates.

    With ``q(eta) = reference * Exp(eta)`` and ``u`` the tool axis in the spatula
    frame, the world tool axis is ``R(q(eta)) u`` and

        d/d eta [R(q(eta)) u] = -R(q(eta)) [u]_x J_r(eta).

    This is what lets orientation enter the geometric constraints of
    :mod:`tplqt.safety` as a decision variable. Accepts ``(3,)`` or ``(T, 3)``.
    """
    from .frames import SPATULA_TOOL_AXIS

    eta = np.asarray(eta, float)
    single = eta.ndim == 1
    eta = eta.reshape(-1, 3)
    rotations = (R.from_quat(reference) * R.from_rotvec(eta)).as_matrix()
    jacobian = -np.einsum("tij,jk,tkl->til", rotations, _skew(SPATULA_TOOL_AXIS),
                          right_jacobian(eta))
    return jacobian[0] if single else jacobian
