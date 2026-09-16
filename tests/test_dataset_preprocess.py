"""Loading recordings, resolving calibrations and preparing strokes.

The synthetic demonstrations are built in the vial frame and pushed out to the
motion-capture frame, so what the preprocessing recovers is known in closed form.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation as R

from conftest import CALIBRATION, synthetic_demonstration, write_demonstration
from tplqt.calibration import (calibration, calibration_name_for_dataset, calibration_names,
                               registry)
from tplqt.dataset import (REQUIRED_FILES, dataset_name, is_demonstration, load_dataset,
                           load_demonstration)
from tplqt.frames import mocap_to_world, vial_bore_axis
from tplqt.preprocess import decimation_factor, in_vial_window, prepare, smooth


def _write_small(tmp_path, name="demo_00", n_samples=40):
    """Write a short demonstration and return its folder."""
    demo = synthetic_demonstration(0, n_samples=n_samples)
    return write_demonstration(str(tmp_path / "set" / name), demo)


def _rewrite(path, edit):
    """Rewrite a JSON-lines stream row by row through ``edit(index, row)``."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(path, "w") as f:
        for i, row in enumerate(rows):
            if edit(i, row) is not None:
                f.write(json.dumps(edit(i, row)) + "\n")


def test_loaded_demonstrations_are_the_ones_that_were_written(written_dataset, demonstrations):
    """Loading a dataset returns the written poses, sorted by folder name."""
    loaded = load_dataset(written_dataset)
    assert [d.name for d in loaded] == sorted(d.name for d in demonstrations)
    assert [d.n_samples for d in loaded] == [600] * len(demonstrations)
    for got, want in zip(loaded, demonstrations):
        assert_allclose(got.spatula_pos, want.spatula_pos, rtol=0, atol=1e-12)
        assert_allclose(got.spatula_quat, want.spatula_quat, rtol=0, atol=1e-12)
        assert_allclose(got.vial_lip_pos, want.vial_lip_pos, rtol=0, atol=1e-12)
        assert_allclose(got.vial_lip_quat, want.vial_lip_quat, rtol=0, atol=1e-12)
        assert got.dataset == "synthetic"


@pytest.mark.parametrize("missing", REQUIRED_FILES)
def test_a_folder_missing_any_required_file_is_not_a_demonstration(tmp_path, missing):
    """Every one of the four recorded files is required to recognise a folder."""
    folder = _write_small(tmp_path)
    assert is_demonstration(folder)
    os.remove(os.path.join(folder, missing))
    assert not is_demonstration(folder)


@pytest.mark.parametrize("kind, message", [("empty", "no demonstration folders"),
                                           ("missing", "not found")])
def test_load_dataset_refuses_a_directory_without_demonstrations(tmp_path, kind, message):
    """A dataset directory that holds no demonstration folders is an error."""
    root = tmp_path / kind
    if kind == "empty":
        (root / "notes").mkdir(parents=True)
    with pytest.raises(ValueError, match=message):
        load_dataset(str(root))


@pytest.mark.parametrize("stream, edit, message", [
    ("vial_lip_in_world.jsonl",
     lambda i, row: {**row, "time": row["time"] + 86400.0}, "common clock"),
    ("force_torque_data_in_bota_frame.jsonl",
     lambda i, row: None if i == 0 else row, "different lengths"),
])
def test_load_demonstration_requires_one_clock_for_the_three_streams(tmp_path, stream,
                                                                     edit, message):
    """Streams that are not sample for sample on one clock are rejected."""
    folder = _write_small(tmp_path)
    _rewrite(os.path.join(folder, stream), edit)
    with pytest.raises(ValueError, match=message):
        load_demonstration(folder)


def test_load_demonstration_rejects_a_vial_that_moved(tmp_path):
    """A vial drifting by more than the static tolerance is rejected."""
    folder = _write_small(tmp_path)
    _rewrite(os.path.join(folder, "vial_lip_in_world.jsonl"),
             lambda i, row: {**row, "position": {**row["position"],
                                                 "x": row["position"]["x"] + 5e-4 * i}})
    with pytest.raises(ValueError, match="vial moved"):
        load_demonstration(folder)


@pytest.mark.parametrize("root", ["/tmp/salt_scoop", "/tmp/salt_scoop/",
                                  "/tmp/salt_scoop//"])
def test_dataset_name_ignores_trailing_separators(root):
    """The dataset label is the directory name, with or without a trailing slash."""
    assert dataset_name(root) == "salt_scoop"


def test_calibration_returns_the_stored_session_transform():
    """A named session resolves to its stored translation and quaternion."""
    translation, quaternion = calibration("2026-06-01")
    assert_allclose(translation, [-0.33923826167657317, 0.40110038382695307,
                                  0.02719264880394548], rtol=0, atol=1e-15)
    assert_allclose(quaternion, [-0.028038625197740173, -0.007405847740145632,
                                 -0.015112212999634845, 0.9994651619413878],
                    rtol=0, atol=1e-15)


def test_an_unknown_calibration_lists_the_known_sessions():
    """Asking for a session that does not exist names the ones that do."""
    with pytest.raises(KeyError) as excinfo:
        calibration("1999-01-01")
    assert all(name in str(excinfo.value) for name in calibration_names())


@pytest.mark.parametrize("dataset", ["salt_scoop", "/data/salt_scoop",
                                     "/data/salt_scoop/"])
def test_a_dataset_path_resolves_to_its_calibration_session(dataset):
    """A dataset is looked up by its directory name, however it is spelled."""
    assert calibration_name_for_dataset(dataset) == "2026-06-01"


def test_an_unregistered_dataset_has_no_calibration():
    """An unregistered dataset raises instead of falling back to a default."""
    with pytest.raises(KeyError, match="has no calibration"):
        calibration_name_for_dataset("/data/not_a_recorded_dataset")


def test_shipped_calibrations_are_unit_quaternions_reachable_from_every_dataset():
    """Every stored quaternion is unit norm and every dataset resolves to one."""
    for name, (translation, quaternion) in registry().items():
        assert translation.shape == (3,) and quaternion.shape == (4,)
        assert float(np.linalg.norm(quaternion)) == pytest.approx(1.0, abs=1e-12)
    for dataset in ("salt_scoop", "honey_scoop", "honey_deposit"):
        assert calibration_name_for_dataset(dataset) in calibration_names()


def _two_dip_profile():
    """A tip that enters the vial twice: samples 10-20 and 60-80 are inside."""
    n = 100
    depth = np.full(n, 0.005)
    depth[10:21] = -0.010
    depth[60:81] = -0.020
    pos = np.zeros((n, 3))
    pos[:, 2] = depth
    return pos, np.arange(n) / 100.0


@pytest.mark.parametrize("contact_index, window", [(15, (10, 20)), (60, (60, 80)),
                                                   (70, (60, 80)), (80, (60, 80))])
def test_the_in_vial_window_is_the_visit_that_contains_the_contact(contact_index, window):
    """Of several visits inside the vial, only the one holding contact is kept."""
    pos, time = _two_dip_profile()
    bore, lip = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    start, end = in_vial_window(pos, time, bore, lip, time[contact_index])
    assert (start, end) == window
    assert pos[start, 2] < 0 and pos[end, 2] < 0
    assert pos[start - 1, 2] > 0 and pos[end + 1, 2] > 0


def test_a_contact_outside_the_vial_is_an_error():
    """Contact recorded above the lip means the frames disagree, and is refused."""
    pos, time = _two_dip_profile()
    with pytest.raises(ValueError, match="outside the vial opening"):
        in_vial_window(pos, time, np.array([0.0, 0.0, 1.0]), np.zeros(3), time[40])


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_smoothing_reproduces_polynomials_up_to_its_order(degree):
    """A polynomial of degree at most the filter order passes through unchanged."""
    t = np.arange(200) / 250.0
    signal = np.stack([0.1 * t ** degree, 0.03 - 0.2 * t ** degree, 0.05 * t ** degree], 1)
    assert_allclose(smooth(signal, 250.0, polyorder=3), signal, rtol=0, atol=1e-12)


def test_a_signal_shorter_than_the_filter_is_returned_unchanged():
    """Too few samples to fit the polynomial leaves the signal untouched."""
    signal = np.arange(12.0).reshape(4, 3)
    smoothed = smooth(signal, 250.0, polyorder=3)
    assert_array_equal(smoothed, signal)
    assert smoothed is not signal


def test_smoothing_attenuates_high_frequency_content():
    """A 60 Hz ripple on a 1 Hz motion is cut to under a tenth of its amplitude."""
    t = np.arange(500) / 250.0
    base = np.stack([0.01 * np.sin(2 * np.pi * t)] * 3, 1)
    ripple = 0.001 * np.stack([np.sin(2 * np.pi * 60.0 * t)] * 3, 1)
    residual = smooth(base + ripple, 250.0) - smooth(base, 250.0)
    assert residual.std() < 0.10 * ripple.std()


@pytest.mark.parametrize("rate_hz, target, factor", [(250.0, 50.0, 5), (250.0, 250.0, 1),
                                                     (100.0, 50.0, 2), (120.0, 50.0, 2),
                                                     (50.0, 250.0, 1), (250.0, 30.0, 8)])
def test_decimation_factor_rounds_to_the_nearest_stride(rate_hz, target, factor):
    """The stride is the rounded rate ratio, never below one."""
    assert decimation_factor(rate_hz, target) == factor


@pytest.mark.parametrize("rate_hz, target, factor", [(180.0, 50.0, 4), (250.0, 90.0, 3),
                                                     (300.0, 80.0, 4), (250.0, 150.0, 2)])
def test_decimation_factor_rounds_the_ratio_instead_of_truncating_it(rate_hz, target,
                                                                     factor):
    """A ratio past the halfway mark rounds up instead of down to the integer below."""
    assert decimation_factor(rate_hz, target) == factor
    assert int(rate_hz / target) == factor - 1


def test_the_wrench_is_windowed_and_downsampled_with_the_positions():
    """The stroke keeps the wrench of exactly the samples its positions are taken from."""
    demo = synthetic_demonstration(3)
    # column 0 of the wrench is the index of the sample it was recorded on
    marked = (np.arange(demo.n_samples, dtype=float)[:, None]
              + np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    demo = dataclasses.replace(demo, wrench=marked)
    stroke = prepare(demo, calibration=CALIBRATION, target_rate=50.0)

    rows = stroke.wrench[:, 0].astype(int)
    assert stroke.wrench.shape == (stroke.n_samples, 6)
    assert_array_equal(stroke.wrench, demo.wrench[rows])
    assert_array_equal(demo.time[rows], stroke.time)
    assert_array_equal(np.diff(rows), np.full(stroke.n_samples - 1, 5))
    # the recorded tip positions of those same rows are what the stroke path smooths
    world = mocap_to_world(demo.spatula_pos[rows], calibration=CALIBRATION)
    assert_allclose(world, stroke.pos_world, rtol=0, atol=1e-6)


def test_a_prepared_stroke_is_the_in_vial_segment_at_the_target_rate():
    """Preparation downsamples to the target rate and keeps only in-vial samples."""
    demo = synthetic_demonstration(3)
    stroke = prepare(demo, calibration=CALIBRATION, target_rate=50.0)
    assert stroke.rate_hz == pytest.approx(50.0, abs=1e-12)
    assert stroke.n_samples == len(stroke.pos_world) == len(stroke.vel_world)
    depth = (stroke.pos_world - demo.vial_lip_pos) @ vial_bore_axis(demo.vial_lip_quat)
    assert depth.max() < 0.0
    # epoch timestamps are float64, so their spacing is exact to about 0.3 us
    assert np.diff(stroke.time) == pytest.approx(1.0 / 50.0, abs=1e-6)


def test_the_contact_sample_and_point_survive_preparation():
    """The contact index is the nearest kept sample and reads as the built contact."""
    contact = np.array([0.006, -0.004, -0.038])
    demo = synthetic_demonstration(3, contact=contact)
    stroke = prepare(demo, calibration=CALIBRATION, target_rate=50.0)
    offsets = np.abs(stroke.time - demo.contact_time)
    assert stroke.contact_index == int(np.argmin(offsets))
    assert offsets[stroke.contact_index] <= 0.5 / stroke.rate_hz
    assert_allclose(stroke.contact_pos_vial, contact, rtol=0, atol=1e-5)


def test_the_vial_path_and_velocity_follow_from_the_world_path():
    """``pos_vial`` is the world path read in the vial pose, ``vel_world`` its derivative."""
    demo = synthetic_demonstration(2)
    stroke = prepare(demo, calibration=CALIBRATION, target_rate=50.0)
    expected = R.from_quat(demo.vial_lip_quat).inv().apply(
        stroke.pos_world - demo.vial_lip_pos)
    assert_allclose(stroke.pos_vial, expected, rtol=0, atol=1e-12)
    central = (stroke.pos_world[2:] - stroke.pos_world[:-2]) * stroke.rate_hz / 2.0
    assert_allclose(stroke.vel_world[1:-1], central, rtol=0, atol=1e-12)


def test_the_radial_convention_turns_only_the_contact_frame():
    """A radial contact frame keeps both origins and points its x axis at the bore."""
    demo = synthetic_demonstration(1)
    bore_aligned = prepare(demo, calibration=CALIBRATION)
    radial = prepare(demo, calibration=CALIBRATION, contact_orientation="radial")
    assert radial.contact_orientation == "radial"
    assert_array_equal(radial.frames["vial"][0], bore_aligned.frames["vial"][0])
    assert_array_equal(radial.frames["vial"][1], bore_aligned.frames["vial"][1])
    assert_array_equal(radial.frames["contact"][1], bore_aligned.frames["contact"][1])
    A_vial = radial.frames["vial"][0]
    A_contact = radial.frames["contact"][0]
    inward = -radial.contact_pos_vial[:2] / np.linalg.norm(radial.contact_pos_vial[:2])
    assert_allclose(A_vial.T @ A_contact[:, 0], [inward[0], inward[1], 0.0],
                    rtol=0, atol=1e-12)
    assert_allclose(A_contact[:, 2], A_vial[:, 2], rtol=0, atol=1e-12)


def test_the_calibration_argument_sets_where_the_stroke_lies_in_world():
    """Shifting the calibration across the bore shifts the stroke by the same vector."""
    demo = synthetic_demonstration(0)
    shift = np.array([0.010, 0.0, 0.0])      # across the bore, so the window is unchanged
    stroke = prepare(demo, calibration=CALIBRATION)
    shifted = prepare(demo, calibration=(CALIBRATION[0] + shift, CALIBRATION[1]))
    assert shifted.n_samples == stroke.n_samples
    assert_allclose(shifted.pos_world - stroke.pos_world,
                    np.tile(shift, (stroke.n_samples, 1)), rtol=0, atol=1e-12)
    A_vial = stroke.frames["vial"][0]
    assert_allclose(shifted.pos_vial - stroke.pos_vial,
                    np.tile(A_vial.T @ shift, (stroke.n_samples, 1)), rtol=0, atol=1e-12)
