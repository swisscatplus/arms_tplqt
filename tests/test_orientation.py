"""Tests for the orientation chart: round trips, anchoring, and derivatives."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from tplqt import orientation as ori
from tplqt.frames import tool_axis_world


def random_quats(n, seed, scale=1.5):
    """``n`` orientations drawn by exponentiating random rotation vectors."""
    rng = np.random.default_rng(seed)
    return R.from_rotvec(rng.normal(scale=scale, size=(n, 3))).as_quat()


def central_difference(f, x, h):
    """Central-difference Jacobian of ``f`` at ``x`` with step ``h``."""
    jacobian = np.zeros((len(f(x)), len(x)))
    for j in range(len(x)):
        step = np.zeros(len(x))
        step[j] = h
        jacobian[:, j] = (f(x + step) - f(x - step)) / (2.0 * h)
    return jacobian


def test_chart_round_trip_recovers_the_orientations():
    """from_chart inverts to_chart for a batch of orientations."""
    quats = random_quats(64, seed=1)
    reference = random_quats(1, seed=2)[0]
    recovered = ori.from_chart(ori.to_chart(quats, reference), reference)
    np.testing.assert_allclose(ori.geodesic_angle_deg(quats, recovered), 0.0, atol=1e-9)


def test_chart_of_the_reference_is_the_origin():
    """The reference orientation sits at the origin of its own chart."""
    for seed in range(4):
        reference = random_quats(1, seed=10 + seed)[0]
        np.testing.assert_allclose(ori.to_chart(reference, reference), np.zeros(3), atol=1e-12)


def test_reference_of_a_symmetric_spread_is_the_central_orientation():
    """The mean of orientations spread symmetrically about one recovers it."""
    central = random_quats(1, seed=3)[0]
    offsets = np.array([[0.30, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.20],
                        [0.15, -0.25, 0.10]])
    spread = np.concatenate([ori.from_chart(offsets, central),
                             ori.from_chart(-offsets, central)])
    mean = ori.reference_orientation(spread)
    assert ori.geodesic_angle_deg(central, mean)[0] == pytest.approx(0.0, abs=1e-6)


def test_reference_reads_a_list_and_a_stacked_array_alike():
    """Pooling demonstrations into a list gives the mean of the stacked array."""
    parts = [random_quats(20, seed=4), random_quats(12, seed=5)]
    from_list = ori.reference_orientation(parts)
    from_array = ori.reference_orientation(np.concatenate(parts))
    assert ori.geodesic_angle_deg(from_list, from_array)[0] == pytest.approx(0.0, abs=1e-9)


def test_world_chart_leaves_the_stored_reference_unchanged():
    """A world-anchored chart uses its reference as stored, whatever the frame."""
    reference = random_quats(1, seed=6)[0]
    turn = R.from_rotvec([0.0, 0.0, 1.1])
    np.testing.assert_array_equal(ori.anchored_reference(reference, "world", turn), reference)
    np.testing.assert_array_equal(ori.anchored_reference(reference, "world", None), reference)


@pytest.mark.parametrize("chart", ["vial", "contact"])
def test_frame_anchored_reference_is_the_frame_rotation_applied(chart):
    """A frame-anchored reference is the stored one rotated by the frame."""
    reference = random_quats(1, seed=7)[0]
    frame = R.from_rotvec([0.2, -0.7, 0.35])
    anchored = ori.anchored_reference(reference, chart, frame)
    expected = (frame * R.from_quat(reference)).as_quat()
    assert ori.geodesic_angle_deg(anchored, expected)[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("chart", ["vial", "contact"])
def test_frame_anchored_reconstruction_turns_with_the_frame(chart):
    """Chart coordinates read in a frame reconstruct an orientation that follows it."""
    reference = random_quats(1, seed=8)[0]
    quat_world = random_quats(1, seed=9)[0]
    frame = R.from_rotvec([0.1, 0.3, -0.5])

    eta = ori.to_chart((frame.inv() * R.from_quat(quat_world)).as_quat(), reference)
    same = ori.from_chart(eta, ori.anchored_reference(reference, chart, frame))
    assert ori.geodesic_angle_deg(quat_world, same)[0] == pytest.approx(0.0, abs=1e-9)

    moved = R.from_rotvec([0.0, 0.0, 0.9]) * frame
    turned = ori.from_chart(eta, ori.anchored_reference(reference, chart, moved))
    expected = (R.from_rotvec([0.0, 0.0, 0.9]) * R.from_quat(quat_world)).as_quat()
    assert ori.geodesic_angle_deg(turned, expected)[0] == pytest.approx(0.0, abs=1e-9)
    assert ori.geodesic_angle_deg(turned, quat_world)[0] == pytest.approx(np.degrees(0.9),
                                                                         abs=1e-9)


@pytest.mark.parametrize("chart", ["vial", "contact"])
def test_frame_anchored_reference_requires_a_frame_rotation(chart):
    """Anchoring in a task frame without that frame's rotation is an error."""
    with pytest.raises(ValueError, match="frame rotation"):
        ori.anchored_reference(random_quats(1, seed=11)[0], chart, None)


def test_right_jacobian_at_the_identity_is_the_identity():
    """The right Jacobian of Exp at zero is the identity matrix."""
    np.testing.assert_allclose(ori.right_jacobian(np.zeros(3)), np.eye(3), atol=1e-15)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 0.3, 1.0])
def test_right_jacobian_matches_the_finite_difference_of_exp(magnitude):
    """Log(Exp(eta)^-1 Exp(eta + d)) equals J_r(eta) d to second order in |d|."""
    rng = np.random.default_rng(12)
    direction = rng.normal(size=3)
    eta = magnitude * direction / np.linalg.norm(direction)
    jacobian = ori.right_jacobian(eta)
    step = 1e-3
    for _ in range(8):
        d = rng.normal(size=3)
        d = step * d / np.linalg.norm(d)
        exact = (R.from_rotvec(eta).inv() * R.from_rotvec(eta + d)).as_rotvec()
        assert np.linalg.norm(exact - jacobian @ d) < 0.5 * step ** 2


def test_right_jacobian_is_not_the_identity_away_from_the_origin():
    """At a one-radian deviation the Jacobian correction is first order, not second."""
    eta = np.array([0.6, -0.5, 0.62])          # |eta| ~ 1 rad
    step = 1e-3
    d = step * np.array([0.0, 0.0, 1.0])
    exact = (R.from_rotvec(eta).inv() * R.from_rotvec(eta + d)).as_rotvec()
    assert np.linalg.norm(exact - d) > 0.05 * step
    assert np.linalg.norm(exact - ori.right_jacobian(eta) @ d) < 0.5 * step ** 2


@pytest.mark.parametrize("seed", [21, 22, 23])
@pytest.mark.parametrize("eta_scale", [0.0, 0.2, 0.8])
def test_tool_axis_jacobian_matches_a_central_difference(seed, eta_scale):
    """The tool-axis Jacobian is the derivative of the world tool axis in the chart."""
    rng = np.random.default_rng(seed)
    reference = R.from_rotvec(rng.normal(size=3)).as_quat()
    eta = eta_scale * rng.normal(size=3)
    numeric = central_difference(
        lambda e: tool_axis_world(ori.from_chart(e, reference)), eta, 1e-5)
    np.testing.assert_allclose(ori.tool_axis_jacobian(eta, reference), numeric, atol=1e-6)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 90.0, 179.0])
def test_geodesic_angle_is_the_rotation_angle(angle_deg):
    """The geodesic angle of a known rotation is that rotation's angle."""
    axis = np.array([1.0, -2.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    base = random_quats(1, seed=13)[0]
    turned = (R.from_quat(base) * R.from_rotvec(np.radians(angle_deg) * axis)).as_quat()
    assert ori.geodesic_angle_deg(base, turned)[0] == pytest.approx(angle_deg, abs=1e-9)


def test_geodesic_angle_is_symmetric_and_vanishes_on_identical_inputs():
    """The geodesic angle does not depend on the order of its arguments."""
    a, b = random_quats(32, seed=14), random_quats(32, seed=15)
    np.testing.assert_allclose(ori.geodesic_angle_deg(a, b), ori.geodesic_angle_deg(b, a),
                               atol=1e-10)
    np.testing.assert_allclose(ori.geodesic_angle_deg(a, a), np.zeros(32), atol=1e-10)


def test_chart_excursion_is_the_largest_deviation_in_degrees():
    """The excursion is the largest chart-coordinate norm, converted to degrees."""
    angles = np.radians([5.0, 12.0, 3.0])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [-1 / np.sqrt(3)] * 3])
    eta = angles[:, None] * axes
    assert ori.chart_excursion_deg(eta) == pytest.approx(12.0, abs=1e-10)
    assert ori.chart_excursion_deg(np.zeros((4, 3))) == pytest.approx(0.0, abs=1e-15)


def test_chart_velocity_of_a_linear_ramp_is_its_slope():
    """Differentiating a ramp in chart coordinates returns the ramp slope."""
    rate_hz = 250.0
    slope = np.array([0.3, -0.2, 0.05])
    time = np.arange(400) / rate_hz
    velocity = ori.chart_velocity(time[:, None] * slope, 1.0 / rate_hz)
    np.testing.assert_allclose(velocity, np.broadcast_to(slope, velocity.shape), atol=1e-10)


def test_smoothing_rejects_noise_and_preserves_a_smooth_signal():
    """Smoothing cuts added high-frequency noise and barely moves a smooth signal."""
    rate_hz = 250.0
    time = np.arange(500) / rate_hz
    clean = np.stack([0.2 * np.sin(2 * np.pi * time),
                      0.1 * np.cos(1.4 * np.pi * time),
                      0.05 * time], axis=1)
    noise = np.random.default_rng(16).normal(scale=0.01, size=clean.shape)

    residual = ori.smooth_chart(clean + noise, rate_hz) - clean
    assert residual.std() < 0.5 * noise.std()
    assert np.abs(ori.smooth_chart(clean, rate_hz) - clean).max() < 1e-5
