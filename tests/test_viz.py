"""Tests for drawing a generated stroke.

The geometry the view is built from -- the vial wireframe and the blade -- is
checked directly, and against the geometry the constraints of :mod:`tplqt.safety`
are written in, so the picture cannot drift away from what is solved.

What is drawn is checked by standing a recorder in for rerun, which keeps these
tests independent of how a recording is stored on disk; that rerun accepts what
the module sends it is checked separately by writing one, and skipped when rerun
is not installed.
"""
from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R

from conftest import vial_pose_of, write_cli_dataset
from tplqt import cli
from tplqt.calibration import load_config
from tplqt.export import save_trajectory
from tplqt.frames import tool_axis_world
from tplqt.safety import (SafetySettings, SpatulaGeometry, VialGeometry, blade_points,
                          containment_margins, wall_radius)
from tplqt.synthesize import synthesize
from tplqt.viz import (PALETTE, Scene, Trajectory, blade_in_spatula_frame, entity_name,
                       log_scene, record_rrd, show, vial_wireframe)

CONTACT = np.array([0.005, -0.003, -0.040])
HORIZON = 30
VIAL = VialGeometry()
SPATULA = SpatulaGeometry()


@pytest.fixture(scope="module")
def generated(model, single_stroke):
    """A short generated stroke to draw."""
    return synthesize(model, vial_pose_of(single_stroke), CONTACT, horizon=HORIZON)


@pytest.fixture(scope="module")
def scene(generated, single_stroke, strokes):
    """That stroke, in the vial it was generated for, over the demonstrations."""
    return Scene(vial_pose=vial_pose_of(single_stroke),
                 trajectories=[Trajectory.from_synthesis(generated, name="stroke")],
                 demonstrations=[s.pos_vial for s in strokes],
                 contact_point=CONTACT, dt=generated.dt, name="synthetic")


def tilted_frame():
    """A vial frame that is neither at the origin nor aligned with the world."""
    A = R.from_rotvec([0.3, -0.2, 0.7]).as_matrix()
    return A, np.array([0.2, -0.1, 0.35])


def rings(strips, n_azimuth=48):
    """The closed cross sections of a wireframe, dropping the lines down the wall."""
    return [s for s in strips if len(s) == n_azimuth]


def cylindrical(points, frame):
    """``(depth, radius)`` of points in a vial frame, depth positive into the vial."""
    A, b = frame
    local = (np.atleast_2d(points) - b) @ A
    return -local[:, 2], np.linalg.norm(local[:, :2], axis=1)


def test_every_ring_has_the_radius_the_constraints_give_the_wall_at_its_depth():
    """The drawn wall is the wall the containment constraints are written against."""
    frame = tilted_frame()
    for ring in rings(vial_wireframe(frame, VIAL)):
        depth, radius = cylindrical(ring, frame)
        assert_allclose(depth, depth[0], atol=1e-12)
        assert_allclose(radius, radius[0], rtol=1e-12)
        expected = (VIAL.lip_radius if depth[0] <= 0
                    else float(wall_radius(-depth[0], VIAL)))
        assert radius[0] == pytest.approx(expected, rel=1e-12)


def test_the_wireframe_spans_the_vial_from_the_lip_to_the_bottom():
    """It is drawn between the lip plane and the bottom, and never outside the wall."""
    frame = tilted_frame()
    strips = vial_wireframe(frame, VIAL)
    depth, radius = cylindrical(np.vstack(strips), frame)
    assert depth.min() == pytest.approx(0.0, abs=1e-12)
    assert depth.max() == pytest.approx(VIAL.length, rel=1e-12)
    assert radius.max() == pytest.approx(VIAL.body_radius, rel=1e-12)
    assert radius.min() == pytest.approx(VIAL.lip_radius, rel=1e-12)


def test_rings_are_closed_and_the_wall_lines_run_the_whole_depth():
    """Each cross section closes on itself; each line down the wall has three corners."""
    strips = vial_wireframe(tilted_frame(), VIAL, n_azimuth=48, n_struts=8)
    cross_sections = rings(strips)
    assert len(cross_sections) == 6
    for ring in cross_sections:
        assert_allclose(ring[0], ring[-1], atol=1e-12)
    walls = [s for s in strips if len(s) == 3]
    assert len(walls) == 8


def test_the_drawn_blade_is_the_rod_the_constraints_keep_inside():
    """Posed by a sample of the stroke, the drawn blade is safety's blade."""
    quat = R.from_rotvec([0.4, 0.1, -0.2]).as_quat()
    tip = np.array([0.1, 0.2, 0.3])
    drawn = tip + blade_in_spatula_frame(SPATULA) @ R.from_quat(quat).as_matrix().T
    expected = blade_points(tip, tool_axis_world(quat), SPATULA.length, [0.0, 1.0])[0]
    assert_allclose(drawn, expected, atol=1e-12)
    assert np.linalg.norm(drawn[1] - drawn[0]) == pytest.approx(SPATULA.length, rel=1e-12)


def test_a_trajectory_must_carry_one_orientation_for_every_position():
    position = np.zeros((5, 3))
    with pytest.raises(ValueError, match="orientation must be"):
        Trajectory(position, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="velocity must be"):
        Trajectory(position, np.zeros((5, 4)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r"position must be \(T, 3\)"):
        Trajectory(np.zeros(3), np.zeros((5, 4)))


def test_a_stroke_without_orientation_cannot_be_drawn(position_model, single_stroke):
    """There is no spatula to pose, so the stroke is refused rather than half drawn."""
    flat = synthesize(position_model, vial_pose_of(single_stroke), CONTACT,
                      horizon=HORIZON)
    with pytest.raises(ValueError, match="no orientation"):
        Trajectory.from_synthesis(flat)


def test_a_scene_holds_the_vial_frame_the_stroke_was_generated_for(scene, single_stroke):
    A, b = scene.frame
    expected_A, expected_b = single_stroke.frames["vial"]
    assert_allclose(A, expected_A, atol=1e-12)
    assert_allclose(b, expected_b, atol=1e-12)
    assert_allclose(scene.to_world(np.zeros(3)), expected_b, atol=1e-12)
    assert_allclose(scene.to_world(CONTACT), expected_A @ CONTACT + expected_b, atol=1e-12)


def test_a_scene_needs_something_to_draw(single_stroke):
    pose = vial_pose_of(single_stroke)
    with pytest.raises(ValueError, match="at least one trajectory"):
        Scene(vial_pose=pose, trajectories=[])
    with pytest.raises(ValueError, match="no samples"):
        Scene(vial_pose=pose,
              trajectories=[Trajectory(np.zeros((0, 3)), np.zeros((0, 4)))])


def test_a_scene_keeps_trajectories_that_can_only_be_walked_once(scene):
    """They are walked once per sample, so a generator would be drawn empty."""
    kept = Scene(vial_pose=scene.vial_pose, trajectories=iter(scene.trajectories))
    assert kept.n_samples == scene.n_samples
    assert len(kept.trajectories) == len(scene.trajectories)


def test_strokes_take_their_colours_from_the_palette_in_turn(scene):
    """Each stroke gets the next pair of colours, and its own choice overrides them."""
    plain = Trajectory(np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    assert scene.colors(0, plain) == PALETTE[0]
    assert scene.colors(1, plain) == PALETTE[1]
    assert scene.colors(len(PALETTE), plain) == PALETTE[0]
    chosen = Trajectory(np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
                        color=(1, 2, 3))
    assert scene.colors(0, chosen) == ((1, 2, 3), PALETTE[0][1])


def test_the_timeline_is_as_long_as_the_longest_stroke(scene, generated):
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (HORIZON // 2, 1))
    short = Trajectory(np.zeros((HORIZON // 2, 3)), quat, name="short")
    assert Scene(vial_pose=scene.vial_pose,
                 trajectories=[short, scene.trajectories[0]]).n_samples == HORIZON


@pytest.mark.parametrize("first,second", [("stroke", "stroke"), ("a b", "a_b")])
def test_two_strokes_of_a_scene_cannot_share_the_name_they_are_drawn_under(scene, first,
                                                                           second):
    """Sharing one would draw them as a single stroke, sample interleaved with sample."""
    position, orientation = scene.trajectories[0].position, scene.trajectories[0].orientation
    with pytest.raises(ValueError, match="come out the same"):
        Scene(vial_pose=scene.vial_pose,
              trajectories=[Trajectory(position, orientation, name=first),
                            Trajectory(position, orientation, name=second)])


def test_a_name_is_drawn_as_one_component_of_an_entity_path():
    assert entity_name("a/b c.npz") == "a_b_c_npz"
    assert entity_name("plain") == "plain"
    assert entity_name("") == "stroke"


class Recorder:
    """A stand-in for rerun that remembers what was logged, and under what path."""

    def __init__(self):
        self.application_id, self.spawned, self.saved = None, False, None
        self.logged, self.stamps = [], []
        for archetype in ("LineStrips3D", "Points3D", "Arrows3D", "Transform3D",
                          "Quaternion"):
            setattr(self, archetype, _archetype(archetype))

    def init(self, application_id, spawn=False):
        self.application_id, self.spawned = application_id, spawn

    def log(self, entity, archetype, static=False):
        self.logged.append((entity, archetype, static))

    def set_time(self, timeline, **stamp):
        self.stamps.append((timeline, stamp))

    def save(self, path):
        self.saved = path

    def entities(self, static=None):
        return [entity for entity, _, is_static in self.logged
                if static is None or is_static == static]

    def archetypes(self, entity):
        return [archetype for logged, archetype, _ in self.logged if logged == entity]


def _archetype(kind):
    """A rerun archetype, remembered as what it is and what it was given."""
    def build(*args, **kwargs):
        return {"kind": kind, "args": args, "kwargs": kwargs}
    return build


@pytest.fixture
def recorder(monkeypatch):
    """Stand a recorder in for rerun, for the duration of one test."""
    fake = Recorder()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    return fake


def test_the_scene_is_drawn_once_and_the_strokes_are_drawn_over_time(recorder, scene):
    """The vial and its context are logged once; only the spatula moves."""
    log_scene(scene)
    static = recorder.entities(static=True)
    assert "vial" in static and "gravity" in static and "contact" in static
    assert [f"demonstrations/{i}" for i in range(len(scene.demonstrations))] == [
        entity for entity in static if entity.startswith("demonstrations/")]
    assert "strokes/stroke/path" in static
    assert "strokes/stroke/spatula/blade" in static
    assert "strokes/stroke/spatula/axes" in static
    assert set(recorder.entities(static=False)) == {"strokes/stroke/spatula",
                                                    "strokes/stroke/velocity"}


def test_every_part_of_the_scene_is_logged_where_it_belongs(recorder, scene):
    """The payloads, not just the paths: a part drawn in the wrong frame is wrong."""
    log_scene(scene)

    def payload(entity):
        return recorder.archetypes(entity)[0]

    # The demonstrations are given in the vial frame and drawn in the vial of the
    # scene, which is neither at the origin nor axis-aligned.
    for index, path_vial in enumerate(scene.demonstrations):
        assert_allclose(payload(f"demonstrations/{index}")["args"][0][0],
                        scene.to_world(path_vial), atol=1e-12)
    assert_allclose(payload("contact")["args"][0][0], scene.to_world(scene.contact_point),
                    atol=1e-12)
    assert_allclose(np.vstack(payload("vial")["args"][0]),
                    np.vstack(vial_wireframe(scene.frame, scene.vial)), atol=1e-12)

    # The stroke is already in world; the blade is in the spatula frame, since the
    # pose logged at each sample is what puts it in the vial.
    assert_allclose(payload("strokes/stroke/path")["args"][0][0],
                    scene.trajectories[0].position, atol=1e-12)
    assert_allclose(payload("strokes/stroke/spatula/blade")["args"][0][0],
                    blade_in_spatula_frame(scene.spatula), atol=1e-12)
    axes = payload("strokes/stroke/spatula/axes")["kwargs"]
    assert_allclose(axes["origins"], [blade_in_spatula_frame(scene.spatula)[1]] * 3,
                    atol=1e-12)
    assert_allclose(np.asarray(axes["vectors"]) / np.linalg.norm(axes["vectors"], axis=1),
                    np.eye(3), atol=1e-12)


def test_the_spatula_is_posed_at_the_stroke_at_every_sample(recorder, scene, generated):
    """Each sample logs the pose of that sample, and the velocity beside it."""
    log_scene(scene)
    poses = recorder.archetypes("strokes/stroke/spatula")
    assert len(poses) == HORIZON
    assert [pose["kind"] for pose in poses] == ["Transform3D"] * HORIZON
    assert_allclose([pose["kwargs"]["translation"] for pose in poses],
                    generated.pos_world, atol=1e-12)
    assert_allclose([pose["kwargs"]["quaternion"]["kwargs"]["xyzw"] for pose in poses],
                    generated.quat_world, atol=1e-12)
    # The velocity is logged outside the spatula, so its pose does not turn it.
    arrows = recorder.archetypes("strokes/stroke/velocity")
    assert len(arrows) == HORIZON
    assert_allclose([arrow["kwargs"]["origins"][0] for arrow in arrows],
                    generated.pos_world, atol=1e-12)


def test_a_stroke_is_drawn_for_as_long_as_it_lasts(recorder, scene):
    """A stroke shorter than the scene stops; the longer one carries the timeline on."""
    short = Trajectory(scene.trajectories[0].position[:HORIZON // 2],
                       scene.trajectories[0].orientation[:HORIZON // 2], name="short")
    log_scene(Scene(vial_pose=scene.vial_pose,
                     trajectories=[short, scene.trajectories[0]], dt=scene.dt))
    assert len(recorder.archetypes("strokes/short/spatula")) == HORIZON // 2
    assert len(recorder.archetypes("strokes/stroke/spatula")) == HORIZON


def test_the_recording_is_stamped_in_samples_and_in_seconds(recorder, scene):
    """Both timelines are laid down, so the stroke can be scrubbed either way."""
    log_scene(scene)
    assert [stamp for stamp in recorder.stamps if stamp[0] == "sample"] == [
        ("sample", {"sequence": t}) for t in range(HORIZON)]
    assert [stamp for stamp in recorder.stamps if stamp[0] == "time"] == [
        ("time", {"duration": t * scene.dt}) for t in range(HORIZON)]


def test_without_a_timestep_the_recording_is_stamped_in_samples_alone(recorder, scene):
    log_scene(Scene(vial_pose=scene.vial_pose, trajectories=scene.trajectories))
    assert {timeline for timeline, _ in recorder.stamps} == {"sample"}


def test_a_stroke_named_for_a_file_still_gives_one_entity(recorder, scene):
    """A name with slashes in it would otherwise open a tree of entities."""
    trajectory = Trajectory(scene.trajectories[0].position,
                            scene.trajectories[0].orientation, name="a/b c.npz")
    log_scene(Scene(vial_pose=scene.vial_pose, trajectories=[trajectory]))
    assert "strokes/a_b_c_npz/path" in recorder.entities()
    assert not any(entity.startswith("strokes/a/") for entity in recorder.entities())


def test_writing_a_recording_names_it_after_the_scene(recorder, scene, tmp_path):
    path = record_rrd(str(tmp_path / "scene.rrd"), scene)
    assert (recorder.application_id, recorder.saved, recorder.spawned) == (
        "tplqt synthetic", path, False)


def test_showing_a_scene_opens_the_viewer_and_writes_nothing(recorder, scene):
    show(scene)
    assert (recorder.spawned, recorder.saved) == (True, None)
    assert "vial" in recorder.entities()


def test_rerun_accepts_the_whole_scene(tmp_path, scene):
    """The one test that really writes a recording, so the calls are known to work.

    Rerun does not raise on a payload it cannot read: it warns and writes the file
    without it. So the warning is made an error, and the recording is required to
    hold more than the header an empty one already has.
    """
    rerun = pytest.importorskip("rerun")
    empty = str(tmp_path / "empty.rrd")
    rerun.init("tplqt empty")
    rerun.save(empty)

    with warnings.catch_warnings():
        warnings.simplefilter("error", rerun.error_utils.RerunWarning)
        path = record_rrd(str(tmp_path / "scene.rrd"), scene)
    assert os.path.getsize(path) > os.path.getsize(empty)


def test_without_rerun_the_error_names_the_package_to_install(monkeypatch, scene,
                                                              tmp_path):
    """The module is imported as rerun but installs as rerun-sdk, so the error says so."""
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match="pip install rerun-sdk"):
        record_rrd(str(tmp_path / "scene.rrd"), scene)


@pytest.fixture
def drawn(monkeypatch):
    """Catch the scene the command line builds instead of recording it."""
    caught = {}

    def capture(path, scene):
        caught["path"], caught["scene"] = path, scene
        return path

    monkeypatch.setattr(cli, "record_rrd", capture)
    return caught


@pytest.fixture(scope="module")
def viewable_dataset(tmp_path_factory):
    """A dataset on disk the view command can overlay, under a registered name."""
    name = sorted(load_config()["datasets"])[0]
    return write_cli_dataset(tmp_path_factory.mktemp("view") / name, 300)


def written_stroke(path, generated, single_stroke, **metadata):
    """A generated stroke on disk, recording the vial pose it was generated for."""
    position, quaternion = vial_pose_of(single_stroke)
    return save_trajectory(str(path), generated, metadata={
        "dataset": "synthetic", "vial_position": position.tolist(),
        "vial_quaternion": quaternion.tolist(),
        "contact_point_vial_frame": CONTACT.tolist(), **metadata})


def test_view_tells_apart_two_files_whose_names_are_drawn_the_same(
        tmp_path, generated, single_stroke, drawn):
    """Basenames that differ only in punctuation still give two strokes, not one."""
    first = written_stroke(tmp_path / "a b.npz", generated, single_stroke)
    second = written_stroke(tmp_path / "a_b.npz", generated, single_stroke)
    cli.main(["view", first, second, "--out", str(tmp_path / "both.rrd")])
    names = [entity_name(t.name) for t in drawn["scene"].trajectories]
    assert len(set(names)) == 2


def test_view_draws_every_trajectory_it_is_given_into_one_recording(
        tmp_path, generated, single_stroke, drawn, capsys):
    """Two strokes are drawn together, and the recording is named after the first."""
    first = written_stroke(tmp_path / "plain.npz", generated, single_stroke)
    second = written_stroke(tmp_path / "safe.npz", generated, single_stroke)
    cli.main(["view", first, second])

    assert drawn["path"] == str(tmp_path / "plain.rrd")
    scene = drawn["scene"]
    assert [t.name for t in scene.trajectories] == ["plain", "safe"]
    assert scene.demonstrations is None
    assert_allclose(scene.contact_point, CONTACT, atol=1e-12)
    position, quaternion = vial_pose_of(single_stroke)
    assert_allclose(scene.to_world(np.zeros(3)), position, atol=1e-12)
    assert_allclose(scene.frame[0], R.from_quat(quaternion).as_matrix(), atol=1e-12)
    assert scene.dt == pytest.approx(generated.dt, rel=1e-12)
    printed = capsys.readouterr().out
    assert first in printed and second in printed
    assert f"rerun {drawn['path']}" in printed


def test_view_measures_each_stroke_against_the_vial_it_draws_it_in(
        tmp_path, generated, single_stroke, drawn, capsys):
    """The printed table says how deep, how close and how much clearance."""
    path = written_stroke(tmp_path / "stroke.npz", generated, single_stroke)
    cli.main(["view", path])

    scene = drawn["scene"]
    wall, lip = containment_margins(scene.trajectories[0].position,
                                    scene.trajectories[0].orientation, scene.frame,
                                    SafetySettings(vial=scene.vial,
                                                   spatula=scene.spatula))
    expected = min(np.min(wall), np.nanmin(lip))
    row = re.search(r"stroke\s+(-?[0-9.]+) mm\s+([0-9.]+) mm\s+([+-][0-9.]+) mm",
                    capsys.readouterr().out)
    assert row is not None
    position = scene.trajectories[0].position
    depth = ((position - scene.frame[1]) @ scene.frame[0][:, 2]).min()
    reach = np.linalg.norm(position - scene.to_world(CONTACT), axis=1).min()
    assert float(row.group(1)) == pytest.approx(depth * 1e3, abs=5e-2)
    assert float(row.group(2)) == pytest.approx(reach * 1e3, abs=5e-3)
    assert float(row.group(3)) == pytest.approx(expected * 1e3, abs=5e-4)


def test_view_puts_the_demonstrations_behind_the_stroke(tmp_path, generated,
                                                        single_stroke, drawn,
                                                        viewable_dataset):
    """The demonstrations are drawn in the vial of the trajectory, not their own."""
    stroke = written_stroke(tmp_path / "stroke.npz", generated, single_stroke)
    out = str(tmp_path / "with_demos.rrd")
    cli.main(["view", stroke, "--demos", viewable_dataset, "--out", out])

    scene = drawn["scene"]
    assert drawn["path"] == out
    assert len(scene.demonstrations) == len(os.listdir(viewable_dataset))
    drawn_in_world = scene.to_world(scene.demonstrations[0])
    A_vial, b_vial = scene.frame
    assert np.linalg.norm((drawn_in_world - b_vial) @ A_vial[:, 0:2], axis=1).max() < 0.05


@pytest.mark.parametrize("changed,complaint", [
    ({"vial_position": [0.5, 0.5, 0.5]}, "vials are"),
    ({"vial_quaternion": list(R.from_rotvec([0.0, 0.0, 0.6]).as_quat())}, "turned"),
    ({"contact_point_vial_frame": [0.01, 0.0, -0.04]}, "contact points are"),
    ({}, "records no"),
])
def test_view_refuses_strokes_of_a_different_situation(tmp_path, generated, single_stroke,
                                                       drawn, changed, complaint):
    """One vial and one contact point, since the whole scene is measured in that frame."""
    first = written_stroke(tmp_path / "first.npz", generated, single_stroke)
    if changed:
        other = written_stroke(tmp_path / "other.npz", generated, single_stroke, **changed)
    else:
        other = save_trajectory(str(tmp_path / "other.npz"), generated)   # no pose at all
    with pytest.raises(SystemExit, match=complaint):
        cli.main(["view", first, other])


def test_view_draws_strokes_of_the_same_situation_together(tmp_path, generated,
                                                           single_stroke, drawn):
    """The poses come from different demonstrations of one vial, so they differ a little."""
    position, quaternion = vial_pose_of(single_stroke)
    first = written_stroke(tmp_path / "first.npz", generated, single_stroke)
    other = written_stroke(tmp_path / "other.npz", generated, single_stroke,
                           vial_position=(position + 1e-5).tolist())
    cli.main(["view", first, other])
    assert len(drawn["scene"].trajectories) == 2


def test_view_needs_a_vial_pose_from_somewhere(tmp_path, generated, single_stroke, drawn):
    """A trajectory that does not record one is refused, and the flag supplies it."""
    path = save_trajectory(str(tmp_path / "bare.npz"), generated)
    with pytest.raises(SystemExit, match="--vial-pose"):
        cli.main(["view", path])

    position, quaternion = vial_pose_of(single_stroke)
    cli.main(["view", path, "--vial-pose", *map(str, [*position, *quaternion])])
    assert_allclose(drawn["scene"].to_world(np.zeros(3)), position, atol=1e-12)
    assert_allclose(drawn["scene"].frame[0], R.from_quat(quaternion).as_matrix(),
                    atol=1e-12)


def test_view_can_open_the_viewer_instead_of_writing_a_recording(recorder, tmp_path,
                                                                 generated,
                                                                 single_stroke):
    """With --spawn and no --out, nothing is written to disk."""
    path = written_stroke(tmp_path / "stroke.npz", generated, single_stroke)
    cli.main(["view", path, "--spawn"])
    assert (recorder.spawned, recorder.saved) == (True, None)
    assert not os.path.exists(tmp_path / "stroke.rrd")
