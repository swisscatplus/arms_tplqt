"""Tests for replaying a demonstration with the model that was fitted on it.

Reproduction is the fidelity measurement: the trajectory is regenerated from the
demonstration's own task frames and state segmentation, so what the error reports
is what the fit lost. The last test is what keeps that number from being a
tautology -- handed another demonstration's task frames, the same machinery is
an order of magnitude worse.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation as R

from tplqt.frames import project_position
from tplqt.model import configuration
from tplqt.reproduce import orientation_rms_deg, position_rms, reproduce

CONTROL_COSTS = (1e-4, 1e4, 1e5, 1e6)


def angle_deg(quat_a, quat_b):
    """Angle in degrees between two orientations, elementwise over a batch."""
    return np.degrees((R.from_quat(quat_a).inv() * R.from_quat(quat_b)).magnitude())


def rms_in_own_vial(stroke, reproduction):
    """Position error of a replay, read in the vial frame of the stroke it replays."""
    in_own_frame = dataclasses.replace(
        reproduction,
        pos_vial=project_position(reproduction.pos_world, *stroke.frames["vial"]))
    return position_rms(stroke, in_own_frame)


def test_the_replay_starts_at_the_demonstrated_configuration_and_velocity(model,
                                                                         single_stroke):
    """The replay starts at the demonstration's own initial configuration and velocity."""
    replay = reproduce(model, single_stroke)
    config, config_vel = configuration(single_stroke, model.q_ref, model.orientation_frame)
    assert replay.xi.shape == (single_stroke.n_samples, 2 * model.n_config)
    assert_array_equal(replay.xi[0], np.concatenate([config[0], config_vel[0]]))
    assert_array_equal(replay.pos_world[0], single_stroke.pos_world[0])
    assert_array_equal(replay.xi[0, model.n_config:model.n_config + 3],
                       single_stroke.vel_world[0])
    assert angle_deg(replay.quat_world[0], single_stroke.quat_world[0]) < 1e-3
    # and not at the state the model would have started from: that one is 10 mm away
    means, _ = model.frame_gaussians(
        model.observation(single_stroke)[1]).sequence(replay.states)
    assert np.linalg.norm(replay.xi[0, :3] - means[0, :3]) > 5e-3


def test_the_state_sequence_is_the_models_own_segmentation(model, single_stroke):
    """The replay is driven by the model's Viterbi segmentation of that demonstration."""
    replay = reproduce(model, single_stroke)
    observation, _ = model.observation(single_stroke)
    assert replay.states.shape == (single_stroke.n_samples,)
    assert_array_equal(replay.states, model.hmm.viterbi(observation))
    assert len(np.unique(replay.states)) == model.n_states


def test_the_contact_state_is_the_one_active_at_the_contact_instant(model, strokes):
    """``contact_state`` is the state the segmentation assigns to the contact sample."""
    for stroke in strokes:
        replay = reproduce(model, stroke)
        assert replay.contact_state == int(replay.states[stroke.contact_index])
        # the contact falls inside the stroke, not at either end of the segmentation
        assert replay.contact_state not in (replay.states[0], replay.states[-1])


def test_the_vial_frame_path_is_the_world_path_read_in_the_vial(model, strokes,
                                                               single_stroke):
    """``pos_vial`` is ``pos_world`` expressed in that demonstration's vial frame."""
    replay = reproduce(model, single_stroke)
    A_vial, b_vial = single_stroke.frames["vial"]
    assert_allclose(replay.pos_vial, project_position(replay.pos_world, A_vial, b_vial),
                    rtol=0, atol=1e-15)
    assert_allclose(replay.pos_vial @ A_vial.T + b_vial, replay.pos_world,
                    rtol=0, atol=1e-12)
    assert np.abs(replay.pos_vial - replay.pos_world).max() > 0.05
    for other in strokes:
        if other.name != single_stroke.name:
            elsewhere = project_position(replay.pos_world, *other.frames["vial"])
            assert np.abs(replay.pos_vial - elsewhere).max() > 1e-3


def test_the_reported_errors_are_those_of_the_returned_trajectory(model, strokes):
    """Both errors recompute from the returned arrays, and are small on these strokes."""
    for stroke in strokes:
        replay = reproduce(model, stroke)
        gap = np.linalg.norm(replay.pos_world - stroke.pos_world, axis=1)
        assert position_rms(stroke, replay) == pytest.approx(
            float(np.sqrt(np.mean(gap ** 2))), rel=1e-9)
        angle = angle_deg(stroke.quat_world, replay.quat_world)
        assert orientation_rms_deg(stroke, replay) == pytest.approx(
            float(np.sqrt(np.mean(angle ** 2))), rel=1e-9)
        # measured at most 1.1 mm and 0.2 deg over the six synthetic demonstrations
        assert 0.0 < position_rms(stroke, replay) < 2e-3
        assert 0.0 < orientation_rms_deg(stroke, replay) < 0.5


def test_a_cheaper_control_tracks_the_demonstration_more_closely(model, single_stroke):
    """Raising the control cost buys smoothness by leaving the demonstration behind."""
    errors = [position_rms(single_stroke,
                           reproduce(model, single_stroke, control_cost=cost))
              for cost in CONTROL_COSTS]
    assert all(cheap < dearer for cheap, dearer in zip(errors, errors[1:]))
    assert errors[0] < 1.5e-3
    assert errors[-1] > 10e-3


def test_a_position_only_replay_has_no_orientation(position_model, single_stroke):
    """A position-only model replays position alone and refuses to report an angle."""
    replay = reproduce(position_model, single_stroke)
    assert replay.quat_world is None and replay.eta is None
    assert replay.xi.shape == (single_stroke.n_samples, 6)
    assert_array_equal(replay.xi[0], np.concatenate([single_stroke.pos_world[0],
                                                     single_stroke.vel_world[0]]))
    assert 0.0 < position_rms(single_stroke, replay) < 2e-3
    with pytest.raises(ValueError, match="no orientation"):
        orientation_rms_deg(single_stroke, replay)


def test_another_demonstrations_task_frames_make_the_replay_much_worse(model, strokes,
                                                                      single_stroke):
    """Wrong task frames give a far worse replay: the error is not the tracker's alone."""
    own = reproduce(model, single_stroke)
    own_rms = rms_in_own_vial(single_stroke, own)
    own_angle = orientation_rms_deg(single_stroke, own)
    assert own_rms < 1.5e-3
    for other in strokes:
        if other.name == single_stroke.name:
            continue
        swapped = dataclasses.replace(single_stroke, frames=other.frames)
        replay = reproduce(model, swapped)
        assert rms_in_own_vial(single_stroke, replay) > max(5e-3, 5.0 * own_rms)
        assert orientation_rms_deg(single_stroke, replay) > 5.0 * own_angle


def test_a_stroke_prepared_the_other_way_is_refused(radial_model, strokes):
    """A model fitted on radial contact frames refuses bore-aligned strokes.

    Read through the wrong convention the contact-frame Gaussians describe a
    different frame, which would inflate the reported error instead of failing.
    """
    with pytest.raises(ValueError, match="contact frames"):
        reproduce(radial_model, strokes[0])
