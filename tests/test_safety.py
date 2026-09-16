"""Tests for the geometric constraints that keep the stroke inside the vial."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import vial_pose_of

from tplqt import orientation as ori
from tplqt.frames import aim_tool_axis
from tplqt.lqt import canonical_system, solve_lqt
from tplqt.model import expand_schedule, lift_task_frame
from tplqt.safety import (InfeasibleTrajectory, SafetySettings, SpatulaGeometry,
                          VialGeometry, blade_points, constrain_to_vial,
                          containment_margins, lip_crossing, wall_radius,
                          worst_violation)
from tplqt.synthesize import task_frames

VIAL, SPATULA = VialGeometry(), SpatulaGeometry()
NARROW = VialGeometry(body_radius=0.006, lip_radius=0.006)
CONTACT = np.array([0.010, 0.0, -0.030])
HORIZON = 55
CONTROL_COST = 1.0


def _narrow(*, max_iterations: int = 2, **overrides) -> SafetySettings:
    """Settings for the 6 mm vial the generated stroke is re-solved for."""
    return SafetySettings(vial=NARROW, max_iterations=max_iterations, **overrides)


def _clearance(problem, xi):
    """Smallest wall and lip clearance of a trajectory in the narrow vial, in metres."""
    quat = ori.from_chart(xi[:, 3:6], problem["q_ref"])
    wall, lip = containment_margins(xi[:, :3], quat, problem["vial_frame"], _narrow())
    return float(np.nanmin(wall)), float(np.nanmin(lip))


def _axial_poses(frame, depth, offset=0.0):
    """Poses whose tip sits at the given depths on the bore and points down it."""
    A, b = frame
    quat = aim_tool_axis(-A[:, 2], np.array([0.0, 0.0, 0.0, 1.0]))
    return (b + np.outer(depth, A[:, 2]) + offset * A[:, 0],
            np.tile(quat, (len(depth), 1)))


def _random_blades(frame, n, seed):
    """Random tip positions and unit tool directions around a vial frame."""
    rng = np.random.default_rng(seed)
    tips = frame[1] + rng.uniform(-0.06, 0.06, (n, 3))
    axes = rng.normal(size=(n, 3))
    return tips, axes / np.linalg.norm(axes, axis=1)[:, None]


@pytest.mark.parametrize("depth, expected", [
    (0.005, np.inf), (0.0, np.inf),
    (-VIAL.shoulder_length, VIAL.body_radius), (-0.05, VIAL.body_radius)])
def test_wall_radius_is_unbounded_above_the_lip_and_constant_below_the_shoulder(
        depth, expected):
    """The wall exists only below the lip and stops widening below the shoulder."""
    assert wall_radius(depth) == expected


def test_wall_radius_tapers_linearly_across_the_shoulder():
    """Across the shoulder the radius interpolates the lip and body radii, elementwise."""
    fraction = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    radius = wall_radius(-fraction * VIAL.shoulder_length)
    assert radius.shape == fraction.shape
    np.testing.assert_allclose(radius, [0.0089, 0.0095, 0.0105, 0.0115, 0.0125],
                               atol=1e-15)


def test_blade_points_subdivide_the_blade_from_the_tip_to_the_base():
    """The samples run from the tip one blade length down the axis, evenly spaced."""
    tip = np.array([[0.02, 0.05, -0.01]])
    axis = np.array([[0.6, -0.8, 0.0]])
    points = blade_points(tip, axis, SPATULA.length, np.linspace(0.0, 1.0, 7))
    np.testing.assert_allclose(points[0, 0], tip[0], atol=1e-15)
    np.testing.assert_allclose(points[0, -1], tip[0] - SPATULA.length * axis[0], atol=1e-15)
    step = np.diff(points[0], axis=0)
    np.testing.assert_allclose(step, np.tile(step[0], (6, 1)), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(step, axis=1), SPATULA.length / 6.0,
                               atol=1e-15)


def test_lip_crossing_activates_exactly_where_the_crossing_lies_along_the_blade(
        single_stroke):
    """The blade straddles the opening exactly when the crossing fraction is in (0, 1)."""
    frame = single_stroke.frames["vial"]
    tips, axes = _random_blades(frame, 4000, seed=3)
    active, fraction, crossing = lip_crossing(tips, axes, frame, SPATULA.length)
    tip_depth = (tips - frame[1]) @ frame[0][:, 2]
    base_depth = (tips - SPATULA.length * axes - frame[1]) @ frame[0][:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = tip_depth / (tip_depth - base_depth)
    np.testing.assert_array_equal(active, (expected > 0.0) & (expected < 1.0))
    assert active.sum() > 500
    np.testing.assert_allclose(fraction[active], expected[active], atol=1e-9, rtol=0.0)
    assert np.isnan(fraction[~active]).all()
    assert np.isfinite(crossing[active]).all()


@pytest.mark.parametrize("slope", [1e-6, 1e-9])
def test_lip_crossing_stays_finite_for_a_nearly_parallel_blade(single_stroke, slope):
    """A blade almost parallel to the lip plane still crosses it at a finite point."""
    A, b = single_stroke.frames["vial"]
    axis = slope * A[:, 2] + np.sqrt(1.0 - slope ** 2) * A[:, 0]
    tip = b + 0.5 * SPATULA.length * slope * A[:, 2] + 0.004 * A[:, 1]
    active, fraction, crossing = lip_crossing(tip, axis, (A, b), SPATULA.length)
    assert bool(active[0])
    assert fraction[0] == pytest.approx(0.5, abs=1e-5)
    assert crossing[0] == pytest.approx(
        np.hypot(0.004, 0.5 * SPATULA.length * np.sqrt(1.0 - slope ** 2)), rel=1e-5)


def test_the_lip_crossing_lies_on_the_lip_plane(single_stroke):
    """The crossing is the point of zero depth, and its radius is measured there."""
    A, b = single_stroke.frames["vial"]
    tips, axes = _random_blades((A, b), 600, seed=11)
    active, fraction, crossing = lip_crossing(tips, axes, (A, b), SPATULA.length)
    point = tips[active] - (fraction[active] * SPATULA.length)[:, None] * axes[active]
    np.testing.assert_allclose((point - b) @ A[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(crossing[active],
                               np.linalg.norm((point - b) @ A[:, 0:2], axis=1), atol=1e-12)


@pytest.mark.parametrize("offset", [0.0, 0.020])
def test_wall_clearance_is_the_wall_radius_less_the_blade_and_its_own_offset(
        single_stroke, offset):
    """A blade below the shoulder clears the body wall by all of it that is unused."""
    frame = single_stroke.frames["vial"]
    tips, quat = _axial_poses(frame, np.linspace(-0.030, -0.016, 8), offset)
    wall, _ = containment_margins(tips, quat, frame, SafetySettings())
    np.testing.assert_allclose(wall, VIAL.body_radius - SPATULA.radius - offset,
                               atol=1e-12)


def test_lip_clearance_is_nan_where_the_blade_does_not_straddle_the_opening(single_stroke):
    """The lip clearance exists exactly at the timesteps where the tip is inside."""
    frame = single_stroke.frames["vial"]
    depth = 0.011 - 0.002 * np.arange(21)
    tips, quat = _axial_poses(frame, depth)
    _, lip = containment_margins(tips, quat, frame, SafetySettings())
    np.testing.assert_array_equal(np.isnan(lip), depth >= 0.0)
    np.testing.assert_allclose(lip[depth < 0.0], VIAL.lip_radius - SPATULA.radius,
                               atol=1e-12)


@pytest.fixture(scope="module")
def plan(model, strokes):
    """An unconstrained stroke for a new contact point, with the tracker's problem data."""
    frames = task_frames(vial_pose_of(strokes[0]), CONTACT, frames=model.frames,
                         contact_orientation=model.contact_orientation)
    lifted = [lift_task_frame(frames[name], with_orientation=True,
                              orientation_frame=model.orientation_frame,
                              n_deriv=model.n_deriv) for name in model.frames]
    product = model.frame_gaussians(lifted)
    states = expand_schedule(model.schedule, HORIZON)
    mu_seq, sigma_seq = product.sequence(states)
    A_dyn, B_dyn = canonical_system(model.n_config, model.n_deriv, model.dt)
    x0 = np.concatenate([frames["vial"][1], product.mu[states[0]][3:6],
                         np.zeros(model.n_config)])
    problem = dict(A_dyn=A_dyn, B_dyn=B_dyn, x0=x0, mu_seq=mu_seq, sigma_seq=sigma_seq,
                   control_cost=CONTROL_COST, vial_frame=frames["vial"],
                   q_ref=model.anchored_reference(frames))
    return problem, solve_lqt(A_dyn, B_dyn, x0, mu_seq, sigma_seq, CONTROL_COST)


@pytest.fixture(scope="module")
def constrained(plan):
    """The same stroke re-solved under the narrow vial's geometry."""
    problem, xi = plan
    return constrain_to_vial(xi, settings=_narrow(), **problem)[0]


def test_only_the_constrained_stroke_fits_the_narrow_vial(plan, constrained):
    """Re-solving puts the blade inside a vial that the tracked stroke goes through."""
    problem, xi = plan
    wall, lip = _clearance(problem, xi)
    assert wall < -5e-4 and lip < -5e-4
    wall, lip = _clearance(problem, constrained)
    assert wall > -1e-5 and lip > -1e-5


def test_the_constrained_stroke_obeys_the_system_dynamics(plan, constrained):
    """The re-solved stroke starts at the given state and is reachable by some control."""
    problem, _ = plan
    np.testing.assert_allclose(constrained[0], problem["x0"], atol=1e-6)
    step = constrained[1:] - constrained[:-1] @ problem["A_dyn"].T
    control = np.linalg.lstsq(problem["B_dyn"], step.T, rcond=None)[0]
    assert np.abs(step - (problem["B_dyn"] @ control).T).max() < 1e-6


def test_a_margin_buys_clearance_from_the_wall(plan, constrained):
    """Solving with a 2 mm margin leaves the blade at least that much further in."""
    problem, xi = plan
    wide = constrain_to_vial(xi, settings=_narrow(margin=0.002), **problem)[0]
    wide_wall, _ = _clearance(problem, wide)
    tight_wall, _ = _clearance(problem, constrained)
    assert wide_wall > tight_wall + 1.5e-3


def test_a_small_orientation_weight_tilts_rather_than_translates(plan):
    """A cheap orientation is spent tilting the blade, a dear one moving it sideways."""
    problem, xi = plan
    soft = constrain_to_vial(xi, settings=_narrow(orientation_weight=0.05), **problem)[0]
    stiff = constrain_to_vial(xi, settings=_narrow(orientation_weight=20.0), **problem)[0]

    def deviation(trajectory, block):
        error = trajectory[:, block] - problem["mu_seq"][:, block]
        return float(np.linalg.norm(error, axis=1).max())

    assert deviation(soft, slice(3, 6)) > 1.3 * deviation(stiff, slice(3, 6))
    assert deviation(soft, slice(0, 3)) < 0.6 * deviation(stiff, slice(0, 3))


def test_worst_violation_measures_how_far_outside_the_blade_reaches(single_stroke):
    """The check the solved stroke is verified against is signed and in metres.

    The constraints are imposed on a linearisation of the spatula direction, so a
    solved stroke is only safe if it is still inside once its own orientation is
    taken into account; this is the quantity that decides that.
    """
    frame = single_stroke.frames["vial"]
    settings = SafetySettings()
    reference = np.array([0.0, 0.0, 0.0, 1.0])
    for offset, outside in [(0.0, False), (0.004, False), (0.020, True)]:
        tips, quat = _axial_poses(frame, np.array([-0.02, -0.03]), offset=offset)
        eta = ori.to_chart(quat, reference)
        xi = np.concatenate([tips, eta], axis=1)
        violation = worst_violation(xi, frame, reference, settings)
        assert (violation > 0.0) is outside
        wall, lip = containment_margins(tips, quat, frame, settings)
        clearances = np.concatenate([wall, lip])
        tightest = clearances[np.isfinite(clearances)].min()
        assert violation == pytest.approx(-tightest, abs=1e-12)


def test_a_vial_narrower_than_the_blade_is_reported_infeasible(plan):
    """A wall inside the blade's own radius admits no stroke and is not passed through."""
    problem, xi = plan
    pinhole = SafetySettings(vial=VialGeometry(body_radius=5e-4, lip_radius=5e-4),
                             max_iterations=2)
    with pytest.raises(InfeasibleTrajectory) as raised:
        constrain_to_vial(xi, settings=pinhole, **problem)
    assert "infeasible" in raised.value.status
    assert raised.value.iteration == 0
    assert raised.value.status in str(raised.value)


def test_the_constrained_solve_is_deterministic(plan, constrained):
    """The same problem solved twice gives the same trajectory to solver tolerance."""
    problem, xi = plan
    again = constrain_to_vial(xi, settings=_narrow(), **problem)[0]
    np.testing.assert_allclose(again, constrained, atol=1e-6)


def test_a_start_outside_the_vial_is_reported_before_solving(plan):
    """A stroke required to start outside the wall has no solution, and says so."""
    problem, xi = plan
    A_vial = problem["vial_frame"][0]
    outside = xi.copy()
    # Below the opening, where the wall exists, and well beyond it.
    outside[:, 0:3] += 0.03 * A_vial[:, 0] - 0.02 * A_vial[:, 2]
    with pytest.raises(InfeasibleTrajectory, match="start"):
        constrain_to_vial(outside, settings=_narrow(),
                          **{**problem, "x0": outside[0]})
