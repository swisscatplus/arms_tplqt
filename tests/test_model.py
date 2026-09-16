"""Tests for the task-parameterised model: frame lifting, observations and fitting."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from tplqt import orientation as ori
from tplqt.gaussian import GaussianMixture
from tplqt.hmm import HiddenMarkovModel
from tplqt.model import (chart_reference, configuration, expand_schedule, fit,
                         lift_task_frame, observation, state_schedule)

FRAME_A = R.from_rotvec([0.2, -0.3, 0.5]).as_matrix()
FRAME_B = np.array([0.1, -0.2, 0.3])
SCENE_ROTATION = R.from_rotvec([0.3, -0.5, 0.9])


def _at_identity(stroke, **changes):
    """A copy of a stroke whose vial frame sits at the identity."""
    return dataclasses.replace(stroke, frames={"vial": (np.eye(3), np.zeros(3))}, **changes)


def _moved(stroke, rotation):
    """A copy of a stroke with vial, contact and spatula all turned by one rotation."""
    frames = {name: (rotation.as_matrix() @ A, rotation.apply(b))
              for name, (A, b) in stroke.frames.items()}
    return dataclasses.replace(stroke, frames=frames,
                               pos_world=rotation.apply(stroke.pos_world),
                               vel_world=rotation.apply(stroke.vel_world),
                               quat_world=(rotation
                                           * R.from_quat(stroke.quat_world)).as_quat())


@pytest.mark.parametrize("orientation_frame, block",
                         [("world", FRAME_A), ("vial", np.eye(3)), ("contact", np.eye(3))])
def test_lifted_chart_block_rotates_only_in_the_world_chart(orientation_frame, block):
    """The chart block of a lifted frame is the frame rotation only when world-anchored."""
    A, b = lift_task_frame((FRAME_A, FRAME_B), with_orientation=True,
                           orientation_frame=orientation_frame)
    np.testing.assert_allclose(A, block_diag(FRAME_A, block, FRAME_A, block), atol=1e-15)
    np.testing.assert_allclose(b, np.concatenate([FRAME_B, np.zeros(9)]), atol=0.0)


def test_lift_without_orientation_is_the_plain_frame_lift():
    """A position-only state lifts to the frame rotation on position and velocity."""
    A, b = lift_task_frame((FRAME_A, FRAME_B), with_orientation=False,
                           orientation_frame="vial")
    np.testing.assert_allclose(A, block_diag(FRAME_A, FRAME_A), atol=1e-15)
    np.testing.assert_allclose(b, np.concatenate([FRAME_B, np.zeros(3)]), atol=0.0)


@pytest.mark.parametrize("fitted_model, n_config", [("model", 6), ("position_model", 3)])
def test_observation_holds_configuration_and_velocity_once_per_frame(request, fitted_model,
                                                                    n_config, single_stroke):
    """An observation is as wide as the number of frames times twice the configuration."""
    fitted = request.getfixturevalue(fitted_model)
    obs, lifted = fitted.observation(single_stroke)
    assert obs.shape == (single_stroke.n_samples, 2 * 2 * n_config)
    assert len(lifted) == 2
    assert all(A.shape == (2 * n_config, 2 * n_config) for A, _ in lifted)


def test_observation_in_an_identity_frame_is_the_state_itself(single_stroke, model):
    """A frame at the identity leaves the state unchanged."""
    stroke = _at_identity(single_stroke)
    obs, _ = observation(stroke, q_ref=model.q_ref, orientation_frame="vial",
                         frames=("vial",))
    state = np.concatenate(configuration(stroke, model.q_ref, "vial"), axis=1)
    np.testing.assert_array_equal(obs, state)


def test_orientation_and_velocity_occupy_separate_observation_blocks(single_stroke, model):
    """Dimensions 3:6 carry the orientation chart and 6:9 the tip velocity."""
    constant = np.repeat(single_stroke.quat_world[:1], single_stroke.n_samples, axis=0)
    stroke = _at_identity(single_stroke, quat_world=constant)
    obs, _ = observation(stroke, q_ref=model.q_ref, orientation_frame="vial",
                         frames=("vial",))
    np.testing.assert_allclose(np.ptp(obs[:, 3:6], axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(obs[:, 9:12], 0.0, atol=1e-10)
    np.testing.assert_array_equal(obs[:, 6:9], stroke.vel_world)
    assert np.ptp(obs[:, 6:9], axis=0).max() > 0.05


def test_configuration_chart_reconstructs_the_world_orientation(single_stroke, model):
    """Position leads the configuration; the chart is read in the vial frame."""
    config, _ = configuration(single_stroke, model.q_ref, "vial")
    np.testing.assert_array_equal(config[:, :3], single_stroke.pos_world)
    anchored = ori.anchored_reference(model.q_ref, "vial",
                                      R.from_matrix(single_stroke.frames["vial"][0]))
    recovered = ori.from_chart(config[:, 3:], anchored)
    assert ori.geodesic_angle_deg(recovered, single_stroke.quat_world).max() < 0.02
    unanchored = ori.from_chart(config[:, 3:], model.q_ref)
    assert ori.geodesic_angle_deg(unanchored, single_stroke.quat_world).min() > 10.0


def test_configuration_without_orientation_is_position_and_velocity(single_stroke):
    """A position-only configuration is the world path and its velocity, unchanged."""
    config, config_vel = configuration(single_stroke, None, "world")
    np.testing.assert_array_equal(config, single_stroke.pos_world)
    np.testing.assert_array_equal(config_vel, single_stroke.vel_world)


@pytest.mark.parametrize("frame", ["vial", "contact"])
def test_anchored_chart_reference_is_invariant_to_moving_the_whole_scene(strokes, frame):
    """A reference read in a task frame does not depend on where the scene stands."""
    moved = [_moved(s, SCENE_ROTATION) for s in strokes]
    angle = ori.geodesic_angle_deg(chart_reference(strokes, frame),
                                   chart_reference(moved, frame))
    assert angle.max() < 1e-6


def test_world_chart_reference_turns_with_the_scene(strokes):
    """The world-anchored reference is not equivariant: it turns by the scene rotation."""
    moved = [_moved(s, SCENE_ROTATION) for s in strokes]
    angle = ori.geodesic_angle_deg(chart_reference(strokes, "world"),
                                   chart_reference(moved, "world")).max()
    assert angle == pytest.approx(np.degrees(SCENE_ROTATION.magnitude()), abs=1e-6)


def test_strokes_with_different_contact_frames_cannot_be_fitted_together(strokes,
                                                                        radial_strokes):
    """Fitting refuses a set of strokes prepared with different contact conventions."""
    with pytest.raises(ValueError, match="disagree"):
        fit(strokes[:2] + radial_strokes[:1], n_states=2)


@pytest.mark.parametrize("prepared_strokes, kwargs, message", [
    ("radial_strokes", {"orientation_frame": "world"}, "radial contact frame"),
    ("strokes", {"orientation_frame": "contact"}, "radial contact frame"),
    ("strokes", {"orientation_frame": "bore"}, "must be one of"),
])
def test_incompatible_chart_and_contact_frame_are_rejected(request, prepared_strokes,
                                                           kwargs, message):
    """A chart that contradicts the contact-frame convention is refused."""
    with pytest.raises(ValueError, match=message):
        fit(request.getfixturevalue(prepared_strokes)[:2], n_states=2, **kwargs)


@pytest.mark.parametrize("fitted_model, expected",
                         [("model", "vial"), ("radial_model", "contact"),
                          ("position_model", "world")])
def test_default_chart_follows_the_contact_convention(request, fitted_model, expected):
    """The chart defaults to the contact frame for radial strokes and the vial otherwise."""
    assert request.getfixturevalue(fitted_model).orientation_frame == expected


@pytest.mark.parametrize("fitted_model, n_config", [("model", 6), ("position_model", 3)])
def test_fitted_model_reports_sizes_consistent_with_the_fit(request, fitted_model,
                                                            n_config, strokes):
    """State sizes, rate and horizon follow from how the model was fitted."""
    fitted = request.getfixturevalue(fitted_model)
    assert fitted.has_orientation is (n_config == 6)
    assert (fitted.n_config, fitted.n_state, fitted.n_states) == (n_config, 2 * n_config, 4)
    assert fitted.hmm.n_dim == 2 * 2 * n_config
    assert fitted.dt == pytest.approx(1.0 / 50.0, abs=1e-12)
    assert fitted.horizon == int(round(np.mean([s.n_samples for s in strokes])))


def test_state_schedule_is_the_order_the_states_are_visited_in(model, strokes):
    """The schedule runs in the order of the motion, from its first state to its last."""
    order = [k for k, _ in model.schedule]
    for stroke in strokes:
        states = model.hmm.viterbi(model.observation(stroke)[0])
        visited = [int(k) for i, k in enumerate(states) if i == 0 or k != states[i - 1]]
        assert visited == order
        assert (int(states[0]), int(states[-1])) == (order[0], order[-1])


def test_relabelling_the_states_relabels_the_schedule_without_reordering_it(model, strokes):
    """The schedule follows the motion rather than the state indices."""
    permutation = np.array([2, 0, 3, 1])
    emissions = model.hmm.emissions
    relabelled = HiddenMarkovModel(
        GaussianMixture(emissions.mu[permutation], emissions.sigma[permutation],
                        emissions.priors[permutation]),
        model.hmm.trans[np.ix_(permutation, permutation)],
        model.hmm.init_priors[permutation])
    observations = [model.observation(s)[0] for s in strokes]
    new_label = np.argsort(permutation)

    schedule = state_schedule(relabelled, observations)
    assert [k for k, _ in schedule] == [int(new_label[k]) for k, _ in model.schedule]
    for (_, expected), (_, got) in zip(model.schedule, schedule):
        assert got == pytest.approx(expected, abs=1e-12)


def test_state_schedule_orders_every_state_once_with_positive_dwell(model):
    """The schedule is a permutation of the states with dwell fractions summing to one."""
    states = [k for k, _ in model.schedule]
    fractions = np.array([f for _, f in model.schedule])
    assert sorted(states) == list(range(model.n_states))
    assert fractions.min() > 0.0
    assert fractions.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("horizon", [4, 5, 9, 60, 137])
def test_expanded_schedule_fills_the_horizon_in_schedule_order(model, horizon):
    """Expansion gives one state per timestep, each state used, in schedule order."""
    order = [k for k, _ in model.schedule]
    expanded = expand_schedule(model.schedule, horizon)
    assert expanded.shape == (horizon,)
    assert sorted(set(expanded.tolist())) == sorted(order)
    ranks = [order.index(int(k)) for k in expanded]
    assert all(later >= earlier for earlier, later in zip(ranks, ranks[1:]))


def test_fitting_the_same_strokes_twice_gives_identical_parameters(strokes):
    """The fit is deterministic: no random initialisation enters it."""
    first, second = fit(strokes, n_states=3), fit(strokes, n_states=3)
    np.testing.assert_array_equal(first.hmm.emissions.mu, second.hmm.emissions.mu)
    np.testing.assert_array_equal(first.hmm.emissions.sigma, second.hmm.emissions.sigma)
    np.testing.assert_array_equal(first.hmm.trans, second.hmm.trans)
    np.testing.assert_array_equal(first.q_ref, second.q_ref)
    assert first.schedule == second.schedule


def test_anchored_reference_reapplies_the_frame_rotation(model, single_stroke):
    """An anchored reference is the stored one turned by the anchoring frame."""
    A = single_stroke.frames["vial"][0]
    anchored = model.anchored_reference(single_stroke.frames)
    np.testing.assert_allclose(R.from_quat(anchored).as_matrix(),
                               A @ R.from_quat(model.q_ref).as_matrix(), atol=1e-12)
    assert ori.geodesic_angle_deg(anchored, model.q_ref).max() > 1.0


def test_world_chart_reference_is_used_unchanged(strokes, single_stroke):
    """A world-anchored chart needs no frame rotation at reconstruction time."""
    world = fit(strokes, n_states=2, orientation_frame="world")
    np.testing.assert_array_equal(world.anchored_reference(single_stroke.frames),
                                  world.q_ref)


def test_frame_gaussians_give_one_world_gaussian_per_state(model, single_stroke):
    """The product over frames is one proper Gaussian per state, in world coordinates."""
    _, lifted = model.observation(single_stroke)
    product = model.frame_gaussians(lifted)
    assert product.mu.shape == (model.n_states, model.n_state)
    assert product.sigma.shape == (model.n_states, model.n_state, model.n_state)
    np.testing.assert_allclose(product.sigma, np.swapaxes(product.sigma, 1, 2), atol=1e-12)
    assert np.linalg.eigvalsh(product.sigma).min() > 0.0
    single = model.frame_gaussians(lifted[:1])
    expected = model.hmm.emissions.marginal(slice(0, model.n_state)).transform(*lifted[0])
    np.testing.assert_array_equal(single.mu, expected.mu)


def test_the_tighter_frame_dominates_the_product(model, single_stroke):
    """A frame with a much smaller covariance pulls the product onto its own mean."""
    _, lifted = model.observation(single_stroke)
    tight_lift = (lifted[1][0] * 1e-2, lifted[1][1])
    loose = model.frame_gaussians([lifted[0]])
    tight = model.frame_gaussians([tight_lift])
    product = model.frame_gaussians([lifted[0], tight_lift])
    separation = float(np.linalg.norm(loose.mu - tight.mu))
    assert separation > 0.1
    assert np.linalg.norm(product.mu - tight.mu) < 0.02 * separation
    assert np.linalg.norm(product.mu - loose.mu) > 0.9 * separation
