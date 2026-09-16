"""Tests for generating a stroke for a new vial pose and contact point."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from conftest import vial_pose

from tplqt import orientation as ori
from tplqt.frames import quat_from_tilt, tool_axis_world
from tplqt.model import expand_schedule, lift_task_frame
from tplqt.safety import SafetySettings
from tplqt.synthesize import (mean_contact, path_rms_by_state, synthesize,
                              task_frames)

HORIZON, SPAN, SHORT = 60, 40, 20
INTEGRATION_STEPS = 10
CONTACT = np.array([0.004, -0.003, -0.040])
POSE = vial_pose(2)


@pytest.fixture(scope="module")
def stroke(model):
    """One generated stroke, started from the orientation the model expects."""
    return synthesize(model, POSE, CONTACT, horizon=HORIZON, tilt=None)


def state_gaussians(model, contact=CONTACT):
    """The model's per-state Gaussians in world, and the chart reference with them."""
    frames = task_frames(POSE, contact, frames=model.frames,
                         contact_orientation=model.contact_orientation)
    lifted = [lift_task_frame(frames[name], with_orientation=model.has_orientation,
                              orientation_frame=model.orientation_frame,
                              n_deriv=model.n_deriv) for name in model.frames]
    return model.frame_gaussians(lifted), model.anchored_reference(frames)


def gravity_angle_deg(quat):
    """Angle in degrees between the tool axis of an orientation and gravity."""
    return np.degrees(np.arccos(np.clip(tool_axis_world(quat) @ [0.0, 0.0, -1.0], -1, 1)))


def reach_gap(model, contact, horizon=HORIZON):
    """Closest a generated stroke comes, in metres, to the contact it was generated for."""
    generated = synthesize(model, POSE, contact, horizon=horizon, tilt=None)
    return float(np.linalg.norm(generated.pos_vial - contact, axis=1).min())


def deep_point(synthesis, n_samples=10):
    """Mean of the deepest samples of a stroke, in the vial frame."""
    return synthesis.pos_vial[np.argsort(synthesis.pos_vial[:, 2])[:n_samples]].mean(axis=0)


def integrate(quat_start, omega_world, dt, start, steps=INTEGRATION_STEPS):
    """Orientation reached by integrating a world angular velocity from a start sample."""
    rotation = R.from_quat(quat_start)
    for t in range(start, start + steps):
        rotation = R.from_rotvec(omega_world[t] * dt) * rotation
    return rotation.as_quat()


def demonstrated_points(model, strokes):
    """Demonstrated tip positions in the vial frame, grouped by the state they decode to."""
    grouped = {}
    for stroke in strokes:
        states = model.hmm.viterbi(model.observation(stroke)[0])
        for state in np.unique(states):
            grouped.setdefault(int(state), []).append(stroke.pos_vial[states == state])
    return {state: np.concatenate(parts) for state, parts in grouped.items()}


def test_task_frames_are_the_vial_pose_and_the_contact_point():
    """The vial frame is the given pose and the contact frame sits at the contact."""
    A_vial, b_vial = task_frames(POSE, CONTACT)["vial"]
    assert_allclose(A_vial, R.from_quat(POSE[1]).as_matrix(), atol=1e-15)
    assert_allclose(b_vial, POSE[0], atol=1e-15)
    for convention in ("vial", "radial"):
        frames = task_frames(POSE, CONTACT, contact_orientation=convention)
        assert_allclose(frames["contact"][1], A_vial @ CONTACT + b_vial, atol=1e-12)


def test_radial_contact_frame_turns_about_the_bore():
    """The radial contact frame keeps the bore as its z axis but turns about it."""
    A_vial = task_frames(POSE, CONTACT)["vial"][0]
    A_bore = task_frames(POSE, CONTACT, contact_orientation="vial")["contact"][0]
    A_radial = task_frames(POSE, CONTACT, contact_orientation="radial")["contact"][0]
    azimuth = np.arctan2(CONTACT[1], CONTACT[0])
    assert_allclose(A_bore, A_vial, atol=1e-15)
    assert_allclose(A_radial[:, 2], A_vial[:, 2], atol=1e-12)
    assert np.linalg.norm(A_radial[:, 0] - A_vial[:, 0]) > 0.5
    assert_allclose(A_vial.T @ A_radial[:, 0],
                    [-np.cos(azimuth), -np.sin(azimuth), 0.0], atol=1e-12)


def test_stroke_starts_at_the_requested_state(model):
    """A given initial state is the first sample of the generated stroke."""
    start = np.concatenate([np.asarray(POSE[0]) + [0.002, -0.001, 0.003],
                            [0.05, -0.02, 0.01], [0.01, 0.0, -0.02], np.zeros(3)])
    generated = synthesize(model, POSE, CONTACT, horizon=SHORT, start=start)
    assert_allclose(generated.xi[0], start, atol=1e-12)


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (0.008, 0.0, 0.0),
                                   (0.0, -0.005, 0.002), (-0.003, 0.004, -0.001)])
def test_default_start_is_the_lip_centre_displaced_in_the_vial_frame(model, shift):
    """By default the stroke starts at the lip centre shifted in the vial frame."""
    generated = synthesize(model, POSE, CONTACT, horizon=SHORT, start_shift=shift)
    assert_allclose(generated.pos_vial[0], shift, atol=1e-9)


@pytest.mark.parametrize("tilt", [55.0, 90.0, 125.0])
def test_default_start_orientation_makes_the_requested_angle_with_gravity(model, tilt):
    """The default start orientation sits at the requested angle from gravity."""
    generated = synthesize(model, POSE, CONTACT, horizon=SHORT, tilt=tilt)
    assert gravity_angle_deg(generated.quat_world[0]) == pytest.approx(tilt, abs=1e-6)


def test_start_orientation_without_a_tilt_is_the_one_the_model_expects(model, stroke):
    """With no tilt the stroke starts at the orientation of its first state."""
    gaussians, _ = state_gaussians(model)
    assert_allclose(stroke.eta[0], gaussians.mu[stroke.states[0]][3:6], atol=1e-12)
    assert abs(gravity_angle_deg(stroke.quat_world[0]) - 90.0) > 10.0


def test_stroke_moves_with_the_vial(model):
    """Moving the vial moves the whole stroke: what the frame-anchored chart is for."""
    A_vial = R.from_quat(POSE[1])
    turned = (A_vial * R.from_rotvec(0.7 * np.array([0.0, 0.0, 1.0]))).as_quat()
    moved = np.asarray(POSE[0]) + np.array([0.05, -0.03, 0.02])
    transform = R.from_quat(turned) * A_vial.inv()

    here = synthesize(model, POSE, CONTACT, horizon=SPAN, tilt=None)
    there = synthesize(model, (moved, turned), CONTACT, horizon=SPAN, tilt=None)
    assert_allclose(there.pos_world,
                    transform.apply(here.pos_world - np.asarray(POSE[0])) + moved,
                    atol=1e-6)
    rotated = (transform * R.from_quat(here.quat_world)).as_quat()
    assert ori.geodesic_angle_deg(rotated, there.quat_world).max() < 1e-3


def test_the_mean_contact_is_the_cluster_centre_and_the_easiest_to_reach(model, strokes):
    """The mean contact centres the recorded ones, and a contact 15 mm out is missed."""
    centre = mean_contact(strokes)
    assert_allclose(centre, np.mean([s.contact_pos_vial for s in strokes], axis=0),
                    rtol=0, atol=1e-15)
    far = centre + np.array([0.015, 0.0, 0.0])
    assert np.linalg.norm(far - centre) == pytest.approx(0.015, abs=1e-15)
    assert reach_gap(model, centre) < 0.002
    assert reach_gap(model, far) > 5.0 * reach_gap(model, centre)


def test_deep_part_of_the_stroke_follows_a_moved_contact_point(model, stroke):
    """The deep part follows a contact moved sideways, without translating the stroke."""
    shift = np.array([-0.012, 0.0, 0.0])
    there = synthesize(model, POSE, CONTACT + shift, horizon=HORIZON, tilt=None)
    direction = shift[:2] / np.linalg.norm(shift[:2])
    assert 0.002 < (deep_point(there)[:2] - deep_point(stroke)[:2]) @ direction < 0.008
    assert np.linalg.norm(there.pos_vial[0] - stroke.pos_vial[0]) < 1e-9


def test_boundary_orientation_is_held_at_the_ends_and_released_in_between(model):
    """A boundary orientation pins entry and exit while the middle departs from it."""
    gaussians, q_ref = state_gaussians(model)
    first = gaussians.mu[expand_schedule(model.schedule, HORIZON)[0]]
    boundary = quat_from_tilt(POSE[1], 75.0, ori.from_chart(first[3:6], q_ref))
    generated = synthesize(model, POSE, CONTACT, horizon=HORIZON,
                           boundary_orientation=boundary)
    angle = ori.geodesic_angle_deg(np.repeat(boundary[None], HORIZON, axis=0),
                                   generated.quat_world)
    middle = angle[HORIZON // 3:2 * HORIZON // 3].max()
    assert angle[0] < 0.01 and angle[-1] < 2.0
    assert middle > 2.0 and middle > 5.0 * max(angle[0], angle[-1])


def test_anchor_end_returns_the_tip_to_the_lip_centre(position_model):
    """Anchoring the end leaves the last sample at the centre of the lip."""
    free = synthesize(position_model, POSE, CONTACT, horizon=HORIZON)
    anchored = synthesize(position_model, POSE, CONTACT, horizon=HORIZON, anchor_end=True)
    assert np.linalg.norm(anchored.pos_world[-1] - np.asarray(POSE[0])) < 1e-3
    assert np.linalg.norm(free.pos_world[-1] - np.asarray(POSE[0])) > 3e-3


def test_the_contact_frame_is_the_one_the_model_was_fitted_with(radial_model):
    """The contact frame follows the model's convention, and no other is accepted."""
    generated = synthesize(radial_model, POSE, CONTACT, horizon=SPAN, tilt=None)
    radial = task_frames(POSE, CONTACT, contact_orientation="radial")["contact"][0]
    assert_allclose(generated.frames["contact"][0], radial, atol=1e-12)
    assert not np.allclose(generated.frames["contact"][0], generated.frames["vial"][0])
    with pytest.raises(ValueError, match="fitted with a 'radial' contact frame"):
        synthesize(radial_model, POSE, CONTACT, horizon=SPAN, tilt=None,
                   contact_orientation="vial")


def test_lean_flips_with_the_side_of_the_vial_the_contact_is_on(radial_model):
    """With the contact-anchored chart the spatula leans towards its own contact."""
    near = synthesize(radial_model, POSE, [0.006, 0.0, -0.030], horizon=SPAN, tilt=None)
    far = synthesize(radial_model, POSE, [-0.006, 0.0, -0.030], horizon=SPAN, tilt=None)
    middle = SPAN // 2
    A_vial = near.frames["vial"][0]
    lean = [(A_vial.T @ tool_axis_world(s.quat_world[middle]))[:2] for s in (near, far)]
    assert min(np.linalg.norm(v) for v in lean) > 5e-3
    assert lean[0] @ lean[1] / np.prod([np.linalg.norm(v) for v in lean]) < -0.9
    assert ori.geodesic_angle_deg(near.quat_world[middle], far.quat_world[middle])[0] > 90


def test_reported_smoothness_and_tracking_match_the_trajectory(model, stroke):
    """The reported jerk and tracking error are those of the returned trajectory."""
    derivative = stroke.pos_world
    for _ in range(3):
        derivative = np.gradient(derivative, stroke.dt, axis=0)
    assert stroke.jerk_rms == pytest.approx(
        float(np.sqrt(np.mean(np.sum(derivative ** 2, axis=1)))), rel=1e-12)
    mu_seq, _ = state_gaussians(model)[0].sequence(stroke.states)
    error = stroke.pos_world - mu_seq[:, :3]
    assert stroke.tracking_rms == pytest.approx(
        float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))), rel=1e-12)


def test_segments_partition_the_horizon(model, stroke):
    """The segments tile the horizon exactly and carry the scheduled states."""
    assert np.array_equal(stroke.states, expand_schedule(model.schedule, HORIZON))
    assert stroke.segments[0].start == 0 and stroke.segments[-1].end == HORIZON - 1
    assert sum(segment.n_samples for segment in stroke.segments) == HORIZON
    for previous, segment in zip(stroke.segments, stroke.segments[1:]):
        assert segment.start == previous.end + 1
    for segment in stroke.segments:
        assert set(stroke.states[segment.start:segment.end + 1]) == {segment.state}


def test_reported_velocity_is_the_velocity_block_of_the_state(model, stroke):
    """The returned velocity is the velocity block of the tracked state."""
    assert np.array_equal(stroke.vel_world,
                          stroke.xi[:, model.n_config:model.n_config + 3])


@pytest.mark.parametrize("start", [10, 20, 30])
def test_angular_velocity_integrates_to_the_generated_orientation(model, stroke, start):
    """The angular velocity integrates to the orientation; the chart rate alone does not."""
    eta_dot = stroke.xi[:, model.n_config + 3:model.n_config + 6]
    without_jacobian = np.einsum("tij,tj->ti",
                                 R.from_quat(stroke.quat_world).as_matrix(), eta_dot)
    end = stroke.quat_world[start + INTEGRATION_STEPS]
    travelled = ori.geodesic_angle_deg(stroke.quat_world[start], end)[0]

    reported = ori.geodesic_angle_deg(
        integrate(stroke.quat_world[start], stroke.omega_world, stroke.dt, start), end)[0]
    chart_rate = ori.geodesic_angle_deg(
        integrate(stroke.quat_world[start], without_jacobian, stroke.dt, start), end)[0]

    assert travelled > 0.5
    assert reported < 0.01 * travelled
    assert chart_rate > 0.02 * travelled


@pytest.mark.parametrize("kwargs", [{"tilt": 72.0},
                                    {"boundary_orientation": [0.0, 0.0, 0.0, 1.0]}])
def test_position_only_model_rejects_orientation_arguments(position_model, kwargs):
    """Asking a position-only model for a start orientation is an error, not a no-op."""
    with pytest.raises(ValueError, match="fitted with orientation"):
        synthesize(position_model, POSE, CONTACT, horizon=SPAN, **kwargs)


def test_position_only_model_rejects_safety_settings(position_model):
    """The geometric constraints need an orientation, so a position model refuses."""
    with pytest.raises(ValueError, match="orientation"):
        synthesize(position_model, POSE, CONTACT, horizon=SHORT, safety=SafetySettings())


def test_path_rms_reports_one_row_per_segment(model, strokes, stroke):
    """The per-state path error has one row per segment and a pooled overall value."""
    rows, overall = path_rms_by_state(model, strokes, stroke)
    assert len(rows) == len(stroke.segments)
    for row, segment in zip(rows, stroke.segments):
        assert set(row) == {"state", "n_generated", "n_demonstrated", "rms",
                            "demonstration_spread"}
        assert (row["state"], row["n_generated"]) == (segment.state, segment.n_samples)
        assert row["n_demonstrated"] > 0
        assert 0.0 <= row["rms"] < 0.05
    assert 0.0 <= overall < 0.05


def test_path_rms_is_the_distance_to_the_demonstrated_points_of_the_same_state(
        model, strokes, stroke):
    """Each row is the nearest distance to its state's points, and their spread."""
    rows, overall = path_rms_by_state(model, strokes, stroke)
    points = demonstrated_points(model, strokes)

    pooled = []
    for row, segment in zip(rows, stroke.segments):
        reference = points[segment.state]
        distance, _ = KDTree(reference).query(
            stroke.pos_vial[segment.start:segment.end + 1])
        pooled.append(distance)
        assert row["n_demonstrated"] == len(reference)
        assert row["rms"] == pytest.approx(float(np.sqrt(np.mean(distance ** 2))), rel=1e-9)
        assert row["demonstration_spread"] == pytest.approx(
            float(np.sqrt(np.trace(np.cov(reference.T, bias=True)))), rel=1e-9)

    assert overall == pytest.approx(
        float(np.sqrt(np.mean(np.concatenate(pooled) ** 2))), rel=1e-9)
