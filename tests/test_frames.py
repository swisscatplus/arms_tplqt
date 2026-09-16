"""Tests for the coordinate transforms and task frames in :mod:`tplqt.frames`."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from conftest import CALIBRATION, vial_pose
from tplqt.frames import (aim_tool_axis, contact_frame, ensure_quaternion_continuity,
                          frame_from_pose, lift_frame, mocap_to_world, project_orientation,
                          project_position, quat_from_tilt, tool_axis_world, vial_bore_axis,
                          world_to_mocap)

TILT_DEG = 18.0            # the tilt the synthetic vial is held at


def rotation_angle(q_a, q_b) -> float:
    """Angle in radians between two orientations."""
    return float((R.from_quat(q_a).inv() * R.from_quat(q_b)).magnitude())


@pytest.fixture
def poses():
    """A batch of motion-capture poses and the single pose taken from its first row."""
    rng = np.random.default_rng(7)
    pos = rng.normal(scale=0.2, size=(8, 3))
    quat = R.from_rotvec(rng.normal(scale=1.0, size=(8, 3))).as_quat()
    return pos, quat


def test_world_to_mocap_inverts_mocap_to_world_for_a_batch(poses):
    """Lifting a batch of poses into world and back reproduces the input."""
    pos, quat = poses
    pos_world, quat_world = mocap_to_world(pos, quat, calibration=CALIBRATION)
    pos_back, quat_back = world_to_mocap(pos_world, quat_world, calibration=CALIBRATION)
    np.testing.assert_allclose(pos_back, pos, atol=1e-12)
    np.testing.assert_allclose(R.from_quat(quat_back).as_matrix(),
                               R.from_quat(quat).as_matrix(), atol=1e-12)


def test_world_to_mocap_inverts_mocap_to_world_for_one_pose(poses):
    """A single pose round trips and keeps its ``(3,)`` shape."""
    pos, quat = poses[0][0], poses[1][0]
    pos_world, quat_world = mocap_to_world(pos, quat, calibration=CALIBRATION)
    assert pos_world.shape == (3,)
    pos_back, quat_back = world_to_mocap(pos_world, quat_world, calibration=CALIBRATION)
    np.testing.assert_allclose(pos_back, pos, atol=1e-12)
    assert rotation_angle(quat_back, quat) == pytest.approx(0.0, abs=1e-12)


def test_mocap_to_world_applies_the_calibration_rotation_and_translation(poses):
    """The lift is ``A p + t`` on positions and a left rotation on orientations."""
    pos, quat = poses
    translation, quat_cal = CALIBRATION
    A_cal = R.from_quat(quat_cal).as_matrix()
    pos_world, quat_world = mocap_to_world(pos, quat, calibration=CALIBRATION)
    np.testing.assert_allclose(pos_world, pos @ A_cal.T + translation, atol=1e-12)
    np.testing.assert_allclose(R.from_quat(quat_world).as_matrix(),
                               A_cal @ R.from_quat(quat).as_matrix(), atol=1e-12)


def test_quaternion_continuity_removes_sign_flips_without_changing_rotations():
    """Sign flips are undone and every underlying rotation is left alone."""
    axis = np.array([0.2, -0.5, 0.8])
    smooth = R.from_rotvec(np.linspace(0.0, 1.5, 40)[:, None] * axis).as_quat()
    flipped = smooth.copy()
    flipped[[3, 4, 5, 17, 31]] *= -1.0
    fixed = ensure_quaternion_continuity(flipped)
    assert np.all(np.sum(fixed[1:] * fixed[:-1], axis=1) > 0)
    np.testing.assert_allclose(R.from_quat(fixed).as_matrix(),
                               R.from_quat(smooth).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(flipped[3], -smooth[3], atol=1e-12)   # input not mutated


def test_quaternion_continuity_leaves_a_continuous_stream_untouched():
    """A stream already in one hemisphere is returned bit for bit."""
    smooth = R.from_rotvec(np.linspace(0.0, 0.9, 25)[:, None] * np.eye(3)[2]).as_quat()
    np.testing.assert_array_equal(ensure_quaternion_continuity(smooth), smooth)


@pytest.mark.parametrize("seed", [0, 1, 3, 5])
def test_bore_axis_is_the_unit_third_column_of_the_vial_frame(seed):
    """The bore axis is the frame's z column, of unit length and independent of spin."""
    position, quat = vial_pose(seed)
    A, b = frame_from_pose(position, quat)
    bore = vial_bore_axis(quat)
    np.testing.assert_allclose(b, position, atol=1e-15)
    np.testing.assert_allclose(bore, A[:, 2], atol=1e-14)
    assert np.linalg.norm(bore) == pytest.approx(1.0, abs=1e-14)
    tilt = np.radians(TILT_DEG)
    np.testing.assert_allclose(bore, [0.0, -np.sin(tilt), np.cos(tilt)], atol=1e-12)


def test_vial_oriented_contact_frame_is_a_pure_translation_of_the_vial_frame():
    """With ``orientation="vial"`` the frame is the vial rotation at the contact point."""
    _, quat = vial_pose(2)
    A_vial = R.from_quat(quat).as_matrix()
    contact_world = np.array([0.31, -0.11, 0.16])
    A, b = contact_frame(A_vial, contact_world, np.array([0.006, -0.004, -0.04]))
    np.testing.assert_array_equal(A, A_vial)
    np.testing.assert_array_equal(b, contact_world)


def test_radial_contact_frame_is_right_handed_with_x_on_the_inward_wall_normal():
    """The radial frame's x axis points at the vial axis and its z axis is the bore."""
    _, quat = vial_pose(3)
    A_vial = R.from_quat(quat).as_matrix()
    contact_vial = np.array([0.0072, -0.0051, -0.038])
    contact_world = A_vial @ contact_vial + np.array([0.3, -0.1, 0.2])
    A, b = contact_frame(A_vial, contact_world, contact_vial, orientation="radial")

    inward_vial = -np.array([contact_vial[0], contact_vial[1], 0.0])
    inward_vial /= np.linalg.norm(inward_vial)
    np.testing.assert_allclose(A[:, 0], A_vial @ inward_vial, atol=1e-12)
    np.testing.assert_allclose(A[:, 2], vial_bore_axis(quat), atol=1e-12)
    np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-12)
    assert np.linalg.det(A) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(b, contact_world)


def test_radial_contact_frame_is_continuous_across_the_azimuth_branch_cut():
    """Contacts either side of the ``atan2`` cut give frames a hair apart."""
    _, quat = vial_pose(1)
    A_vial = R.from_quat(quat).as_matrix()
    eps, radius = 1e-6, 0.008
    frames = [contact_frame(A_vial, np.zeros(3),
                            [radius * np.cos(t), radius * np.sin(t), -0.04],
                            orientation="radial")[0] for t in (np.pi - eps, -np.pi + eps)]
    angle = float(R.from_matrix(frames[0].T @ frames[1]).magnitude())
    assert angle == pytest.approx(2 * eps, rel=1e-6)


def test_radial_contact_frame_turns_with_the_contact_azimuth():
    """A quarter turn of the contact about the bore turns the frame a quarter turn."""
    _, quat = vial_pose(0)
    A_vial = R.from_quat(quat).as_matrix()
    A_a = contact_frame(A_vial, np.zeros(3), [0.008, 0.0, -0.04], orientation="radial")[0]
    A_b = contact_frame(A_vial, np.zeros(3), [0.0, 0.008, -0.04], orientation="radial")[0]
    np.testing.assert_allclose(R.from_matrix(A_a.T @ A_b).as_rotvec(),
                               [0.0, 0.0, np.pi / 2], atol=1e-12)


def test_a_contact_on_the_vial_axis_has_no_radial_frame():
    """Inside ``radial_tol`` the wall normal is undefined, so the frame is refused."""
    A_vial = R.from_quat(vial_pose(0)[1]).as_matrix()
    with pytest.raises(ValueError, match="no wall normal"):
        contact_frame(A_vial, np.zeros(3), [4e-4, -3e-4, -0.04],
                      orientation="radial", radial_tol=1e-3)
    with warnings.catch_warnings():                      # just outside the tolerance
        warnings.simplefilter("error")
        A_out, _ = contact_frame(A_vial, np.zeros(3), [1.1e-3, 0.0, -0.04],
                                 orientation="radial", radial_tol=1e-3)
    assert not np.allclose(A_out, A_vial)


def test_unknown_contact_orientation_is_rejected():
    """Only the documented orientation conventions are accepted."""
    with pytest.raises(ValueError, match="contact orientation must be one of"):
        contact_frame(np.eye(3), np.zeros(3), np.array([0.01, 0.0, -0.04]),
                      orientation="tangential")


def test_tool_axis_is_the_body_negative_y_axis_in_world(poses):
    """The spatula points along the body's -y axis, for one pose and for a batch."""
    quat = poses[1]
    matrices = R.from_quat(quat).as_matrix()
    np.testing.assert_allclose(tool_axis_world(quat), -matrices[:, :, 1], atol=1e-14)
    np.testing.assert_allclose(tool_axis_world(quat[0]), -matrices[0][:, 1], atol=1e-14)


@pytest.mark.parametrize("tilt_deg", [20.0, 45.0, 70.0, 90.0, 115.0])
def test_tilt_places_the_tool_axis_at_the_requested_angle_from_gravity(tilt_deg):
    """The tool axis makes ``tilt_deg`` with gravity and leans into the vial."""
    _, quat = vial_pose(2)
    reference = R.from_rotvec([0.3, -0.2, 0.1]).as_quat()
    tool = tool_axis_world(quat_from_tilt(quat, tilt_deg, reference))
    inward = -vial_bore_axis(quat)
    heading = inward - inward[2] * np.array([0.0, 0.0, 1.0])
    heading /= np.linalg.norm(heading)
    tilt = np.radians(tilt_deg)
    assert np.linalg.norm(tool) == pytest.approx(1.0, abs=1e-12)
    assert tool @ np.array([0.0, 0.0, -1.0]) == pytest.approx(np.cos(tilt), abs=1e-12)
    assert tool @ heading == pytest.approx(np.sin(tilt), abs=1e-12)
    if tilt_deg == 90.0:
        assert tool[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("tilt_deg", [20.0, 60.0, 90.0, 140.0])
def test_an_upright_vial_takes_its_heading_from_the_reference_orientation(tilt_deg):
    """A vertical bore has no heading of its own, so the reference's own heading is used."""
    upright = np.array([0.0, 0.0, 0.0, 1.0])
    reference = R.from_rotvec([0.3, -0.2, 0.1]).as_quat()
    heading = tool_axis_world(reference) * np.array([1.0, 1.0, 0.0])
    heading /= np.linalg.norm(heading)
    tool = tool_axis_world(quat_from_tilt(upright, tilt_deg, reference))
    tilt = np.radians(tilt_deg)
    assert np.linalg.norm(vial_bore_axis(upright)[:2]) == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.norm(tool) == pytest.approx(1.0, abs=1e-12)
    assert tool @ np.array([0.0, 0.0, -1.0]) == pytest.approx(np.cos(tilt), abs=1e-12)
    assert tool @ heading == pytest.approx(np.sin(tilt), abs=1e-12)


@pytest.mark.parametrize("target", [[0.0, 0.0, -1.0], [0.3, -0.5, 0.2], None])
def test_aiming_the_tool_axis_is_exact_and_minimal(target):
    """The aimed tool axis lands on the target by the shortest rotation."""
    reference = R.from_rotvec([0.2, 0.7, -0.4]).as_quat()
    current = tool_axis_world(reference)
    target = -current if target is None else np.array(target) / np.linalg.norm(target)
    aimed = aim_tool_axis(target, reference)
    np.testing.assert_allclose(tool_axis_world(aimed), target, atol=1e-12)
    expected = float(np.arccos(np.clip(current @ target, -1.0, 1.0)))
    assert rotation_angle(reference, aimed) == pytest.approx(expected, abs=1e-9)


def test_projection_into_a_frame_inverts_placement_into_world():
    """Reading a pose in a frame undoes writing it out of that frame."""
    A, b = frame_from_pose(*vial_pose(4))
    pos_frame = np.array([[0.006, -0.004, -0.04], [0.0, 0.01, -0.02], [-0.008, 0.002, 0.003]])
    np.testing.assert_allclose(project_position(pos_frame @ A.T + b, A, b),
                               pos_frame, atol=1e-12)
    single = project_position(A @ pos_frame[0] + b, A, b)
    assert single.shape == (1, 3)
    np.testing.assert_allclose(single[0], pos_frame[0], atol=1e-12)

    frame_quat = R.from_matrix(A).as_quat()
    quat_frame = R.from_rotvec([0.4, -0.1, 0.9]).as_quat()
    world = (R.from_quat(frame_quat) * R.from_quat(quat_frame)).as_quat()
    assert rotation_angle(project_orientation(world, frame_quat), quat_frame) == pytest.approx(
        0.0, abs=1e-12)


@pytest.mark.parametrize("n_deriv", [1, 2, 3])
def test_lifted_frame_is_block_diagonal_with_only_the_first_block_translated(n_deriv):
    """The lifted frame is ``blockdiag(A, ...)`` with the offset ``[b, 0, ...]``."""
    A, b = frame_from_pose(*vial_pose(5))
    A_lift, b_lift = lift_frame(A, b, n_deriv=n_deriv)
    assert A_lift.shape == (3 * n_deriv, 3 * n_deriv)
    for i in range(n_deriv):
        for j in range(n_deriv):
            expected = A if i == j else np.zeros((3, 3))
            np.testing.assert_allclose(A_lift[3 * i:3 * i + 3, 3 * j:3 * j + 3],
                                       expected, atol=1e-15)
    np.testing.assert_allclose(b_lift[:3], b, atol=1e-15)
    np.testing.assert_allclose(b_lift[3:], 0.0, atol=1e-15)


def test_lifted_frame_translates_position_and_only_rotates_velocity():
    """A lifted state maps as the frame applied to position and to velocity separately."""
    A, b = frame_from_pose(*vial_pose(2))
    pos_frame = np.array([0.006, -0.004, -0.04])
    vel_frame = np.array([0.02, 0.15, -0.3])
    A_lift, b_lift = lift_frame(A, b, n_deriv=2)
    world = A_lift @ np.concatenate([pos_frame, vel_frame]) + b_lift
    np.testing.assert_allclose(world[:3], A @ pos_frame + b, atol=1e-12)
    np.testing.assert_allclose(world[3:], A @ vel_frame, atol=1e-12)
