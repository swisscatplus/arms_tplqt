"""Coordinate transforms and the task frames of the stroke model.

A task frame is a pair ``(A, b)``: ``A`` is the rotation whose columns are the
frame axes expressed in world, and ``b`` is the frame origin in world. A world
point is read in that frame as ``x_frame = A.T (p_world - b)``, and a frame point
is placed in world as ``p_world = A x_frame + b``.

Two frames parameterise a scoop:

vial frame
    the vial-lip pose, its z axis along the bore and pointing out of the opening,
    its origin at the centre of the lip;
contact frame
    the point where the spatula meets the material. Its orientation is either the
    vial frame's, which makes it a pure translation of the vial frame, or radial,
    with its x axis on the inward wall normal so that the frame turns with the
    contact's azimuth.

The spatula is described by its tool axis, the direction it points in its own body
frame; :func:`quat_from_tilt` and :func:`aim_tool_axis` build orientations that aim
it somewhere without disturbing the roll around it.

Quaternions are xyzw throughout.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# The spatula rigid body has its origin at the tool tip and its y axis running from
# the tip to the base, so the tool points along the body's negative y axis.
SPATULA_TOOL_AXIS = np.array([0.0, -1.0, 0.0])

CONTACT_ORIENTATIONS = ("vial", "radial")

Frame = Tuple[np.ndarray, np.ndarray]


def mocap_to_world(pos_mocap, quat_mocap=None, *, calibration):
    """Lift motion-capture poses into world.

    ``calibration`` is the ``(translation, quaternion)`` pair of the recording
    session (see :mod:`tplqt.calibration`), giving
    ``p_world = R(q) p_mocap + t`` and ``q_world = q * q_mocap``. Positions may be
    a single ``(3,)`` point or a ``(N, 3)`` batch.
    """
    t, q = np.asarray(calibration[0], float), np.asarray(calibration[1], float)
    rotation = R.from_quat(q)
    pos = np.asarray(pos_mocap, float)
    pos_world = (rotation.apply(pos.reshape(-1, 3)) + t).reshape(pos.shape)
    if quat_mocap is None:
        return pos_world
    return pos_world, (rotation * R.from_quat(quat_mocap)).as_quat()


def world_to_mocap(pos_world, quat_world=None, *, calibration):
    """Inverse of :func:`mocap_to_world`."""
    t, q = np.asarray(calibration[0], float), np.asarray(calibration[1], float)
    rotation = R.from_quat(q).inv()
    pos = np.asarray(pos_world, float)
    pos_mocap = rotation.apply(pos.reshape(-1, 3) - t).reshape(pos.shape)
    if quat_world is None:
        return pos_mocap
    return pos_mocap, (rotation * R.from_quat(quat_world)).as_quat()


def ensure_quaternion_continuity(quats) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in the same hemisphere.

    A rotation has two quaternion representations, ``q`` and ``-q``; a recorded
    stream can jump between them, which would show up as a discontinuity in any
    quantity computed sample by sample.
    """
    q = np.array(quats, float, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def frame_from_pose(pos_world, quat_world) -> Frame:
    """The task frame ``(A, b)`` of a world pose."""
    return R.from_quat(quat_world).as_matrix(), np.asarray(pos_world, float)


def vial_bore_axis(vial_lip_quat) -> np.ndarray:
    """The bore axis in world: the vial frame's z axis, pointing out of the vial."""
    return R.from_quat(vial_lip_quat).as_matrix()[:, 2]


def contact_frame(A_vial, contact_world, contact_pos_vial, *,
                  orientation: str = "vial", radial_tol: float = 1e-3) -> Frame:
    """The contact task frame, with its origin at the contact point.

    With ``orientation="vial"`` the frame copies the vial frame's orientation, so
    it is a pure translation of it -- the natural choice when the motion is
    organised by depth along the bore, as in scooping.

    With ``orientation="radial"`` the frame is turned about the bore by the
    contact's azimuth ``theta = atan2(y, x)`` measured in the vial frame,
    ``A_contact = A_vial Rz(theta + pi)``, so its x axis is the inward wall normal
    (pointing at the vial axis), its y axis is tangential and its z axis is still
    the bore. A motion learned in this frame generalises around the wall instead
    of being tied to one side of the vial, which is what a deposit against the
    wall needs. ``Rz`` is periodic in the azimuth, so the frame is a continuous
    function of the contact point despite the branch cut of ``atan2``.

    The azimuth is undefined for a contact on the vial axis, so a contact closer
    than ``radial_tol`` to it is refused rather than resolved to an arbitrary
    direction. A contact against the wall, which is what the radial frame is for,
    is far from that.
    """
    if orientation not in CONTACT_ORIENTATIONS:
        raise ValueError(f"contact orientation must be one of {CONTACT_ORIENTATIONS}, "
                         f"got {orientation!r}")
    A_vial = np.asarray(A_vial, float)
    origin = np.asarray(contact_world, float)
    if orientation == "vial":
        return A_vial, origin

    contact = np.asarray(contact_pos_vial, float)
    radius = float(np.hypot(contact[0], contact[1]))
    if radius < radial_tol:
        raise ValueError(
            f"the contact point is {radius * 1e3:.2f} mm from the vial axis, within the "
            f"{radial_tol * 1e3:.1f} mm tolerance, so there is no wall normal to point the "
            "radial contact frame along; use the vial orientation for a contact this close "
            "to the axis")
    azimuth = float(np.arctan2(contact[1], contact[0]))
    turn = R.from_rotvec((azimuth + np.pi) * np.array([0.0, 0.0, 1.0])).as_matrix()
    return A_vial @ turn, origin


def tool_axis_world(spatula_quat_world) -> np.ndarray:
    """Direction the spatula points in world, for one pose or a ``(N, 4)`` batch."""
    return R.from_quat(spatula_quat_world).apply(SPATULA_TOOL_AXIS)


def _minimal_rotation(a, b) -> np.ndarray:
    """Rotation taking unit vector ``a`` to unit vector ``b`` along the shortest arc."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    axis = np.cross(a, b)
    sin = float(np.linalg.norm(axis))
    cos = float(a @ b)
    if sin < 1e-12:
        if cos > 0.0:
            return np.eye(3)
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(a, seed)
        perpendicular /= np.linalg.norm(perpendicular)
        return R.from_rotvec(np.pi * perpendicular).as_matrix()
    return R.from_rotvec(np.arctan2(sin, cos) * (axis / sin)).as_matrix()


def aim_tool_axis(direction_world, reference_quat) -> np.ndarray:
    """Orientation whose tool axis points along ``direction_world``.

    The roll about the tool axis is inherited from ``reference_quat`` through the
    shortest rotation that re-aims it, so the scoop face keeps the orientation it
    has in the demonstrations instead of an arbitrary one.
    """
    target = np.asarray(direction_world, float)
    target = target / np.linalg.norm(target)
    current = tool_axis_world(reference_quat)
    current = current / np.linalg.norm(current)
    realign = _minimal_rotation(current, target)
    return (R.from_matrix(realign) * R.from_quat(reference_quat)).as_quat()


def quat_from_tilt(vial_quat, tilt_deg: float, reference_quat) -> np.ndarray:
    """Orientation whose tool axis sits ``tilt_deg`` from gravity, aimed into the vial.

    The tool axis is placed in the vertical plane that contains the vial's
    horizontal heading, so ``tilt_deg = 90`` points the spatula horizontally into
    the vial, a smaller angle tips it down into the vial and a larger one up
    towards the opening. Roll comes from ``reference_quat`` as in
    :func:`aim_tool_axis`.
    """
    gravity = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 0.0, 1.0])
    inward = -vial_bore_axis(vial_quat)
    heading = inward - (inward @ up) * up
    norm = float(np.linalg.norm(heading))
    if norm < 1e-9:                       # an upright vial has no horizontal heading
        tool = tool_axis_world(reference_quat)
        heading = tool - (tool @ up) * up
        norm = float(np.linalg.norm(heading))
        if norm < 1e-9:
            heading, norm = np.array([1.0, 0.0, 0.0]), 1.0
    tilt = np.radians(float(tilt_deg))
    target = np.cos(tilt) * gravity + np.sin(tilt) * (heading / norm)
    return aim_tool_axis(target / np.linalg.norm(target), reference_quat)


def project_position(pos_world, A, b) -> np.ndarray:
    """Express world position(s) in the frame ``(A, b)``; always returns ``(N, 3)``."""
    return (np.atleast_2d(np.asarray(pos_world, float)) - b) @ A


def project_orientation(quat_world, frame_quat) -> np.ndarray:
    """Express world orientation(s) in the frame given by ``frame_quat``."""
    return (R.from_quat(frame_quat).inv() * R.from_quat(quat_world)).as_quat()


def lift_frame(A, b, n_deriv: int = 2) -> Frame:
    """Extend a frame to a state that stacks a configuration and its derivatives.

    Derivatives rotate with the frame but do not translate, so the lifted frame is
    ``blockdiag(A, ..., A)`` with the offset ``[b, 0, ..., 0]``.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    offset = np.concatenate([b, np.zeros(A.shape[0] * (n_deriv - 1))])
    return np.kron(np.eye(n_deriv), A), offset
