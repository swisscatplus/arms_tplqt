"""Tests on a recorded dataset, run only when one is available.

They pin what the package reaches on real demonstrations rather than on the
synthetic strokes the rest of the suite uses, and are written to hold for any of
the recorded datasets. Point ``TPLQT_DATASET`` at one to run them::

    TPLQT_DATASET=path/to/dataset pytest tests/test_recorded_dataset.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import tplqt
from tplqt import orientation as ori

STATES = 5
DEMONSTRATIONS = 8
CONSTRAINED_HORIZON = 90
TILT_DEG = 72.0


@pytest.fixture(scope="module")
def recorded(recorded_strokes):
    strokes = list(recorded_strokes)[:DEMONSTRATIONS]
    return strokes, tplqt.fit(strokes, n_states=STATES)


@pytest.fixture(scope="module")
def contact(recorded):
    """The centre of the contact points the demonstrations recorded."""
    strokes, _ = recorded
    return tplqt.mean_contact(strokes)


@pytest.fixture(scope="module")
def situation(recorded):
    strokes, _ = recorded
    A_vial, b_vial = strokes[0].frames["vial"]
    return (b_vial, R.from_matrix(A_vial).as_quat()), (A_vial, b_vial)


@pytest.fixture(scope="module")
def generated(recorded, situation, contact):
    """A stroke generated for a demonstrated vial pose, at the model's own timing."""
    _, model = recorded
    vial_pose, _ = situation
    return tplqt.synthesize(model, vial_pose, contact, tilt=TILT_DEG)


def test_every_stroke_is_the_part_of_the_recording_inside_the_vial(recorded):
    """Each prepared stroke lies at the opening or below it and reaches well inside.

    The window is chosen on the recorded positions and the smoothing is applied
    after it, so a sample at the boundary can end a few tens of micrometres above
    the lip; the stroke as a whole is inside.
    """
    strokes, _ = recorded
    for stroke in strokes:
        assert stroke.pos_vial[:, 2].max() < 5e-4
        assert stroke.pos_vial[:, 2].min() < -0.005
        assert 0 <= stroke.contact_index < stroke.n_samples


def test_reproduction_error_is_millimetric(recorded):
    """Replaying the demonstrations stays within a few millimetres and degrees."""
    strokes, model = recorded
    position, orientation = [], []
    for stroke in strokes:
        replay = tplqt.reproduce(model, stroke)
        position.append(tplqt.position_rms(stroke, replay))
        orientation.append(tplqt.orientation_rms_deg(stroke, replay))
    assert np.mean(position) < 0.008
    assert np.mean(orientation) < 10.0


def test_the_chart_stays_in_its_small_angle_regime(recorded):
    """The demonstrated orientations stay near the chart reference."""
    strokes, model = recorded
    excursions = []
    for stroke in strokes:
        A_frame = stroke.frames[model.orientation_frame][0]
        relative = (R.from_matrix(A_frame).inv() * R.from_quat(stroke.quat_world)).as_quat()
        excursions.append(ori.chart_excursion_deg(ori.to_chart(relative, model.q_ref)))
    assert max(excursions) < 45.0


def test_generated_stroke_reaches_the_contact_point(situation, generated, contact):
    """A stroke generated for a demonstrated vial pose reaches the contact point."""
    _, (A_vial, b_vial) = situation
    target = A_vial @ contact + b_vial
    assert np.linalg.norm(generated.pos_world - target, axis=1).min() < 0.005
    assert generated.pos_vial[:, 2].min() < 0.8 * contact[2]


def test_generated_stroke_stays_on_the_demonstrated_manifold(recorded, generated):
    """The generated stroke stays closer to the demonstrations than they spread."""
    strokes, model = recorded
    rows, overall = tplqt.path_rms_by_state(model, strokes, generated)
    spread = np.mean([row["demonstration_spread"] for row in rows])
    assert overall < spread


def test_the_constrained_stroke_is_contained(recorded, situation, contact):
    """The constrained stroke keeps the blade inside the vial it was solved for."""
    _, model = recorded
    vial_pose, vial_frame = situation
    settings = tplqt.SafetySettings(max_iterations=2)
    # A target well inside the wall, so the stroke is shaped by the constraints
    # rather than simply blocked by them.
    target = np.array([0.4 * settings.vial.body_radius, 0.0, contact[2]])
    stroke = tplqt.synthesize(model, vial_pose, target, horizon=CONSTRAINED_HORIZON,
                              tilt=TILT_DEG, safety=settings)
    wall, lip = tplqt.containment_margins(stroke.pos_world, stroke.quat_world,
                                          vial_frame, settings)
    assert np.nanmin(wall) > -settings.containment_tolerance
    assert np.nanmin(lip) > -settings.containment_tolerance
