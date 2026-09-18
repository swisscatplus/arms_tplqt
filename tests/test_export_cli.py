"""Tests for writing a generated stroke to disk and for the command line.

The command line reads a dataset directory and looks its calibration up by
directory name, so the demonstrations are written here under a name the packaged
registry knows, re-expressed in the motion-capture frame of that calibration.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conftest import vial_pose_of, write_cli_dataset
from tplqt import cli
from tplqt.calibration import load_config
from tplqt.dataset import load_dataset
from tplqt.export import FIELDS, load_trajectory, save_trajectory
from tplqt.preprocess import prepare_dataset
from tplqt.safety import SafetySettings, VialGeometry, containment_margins
from tplqt.synthesize import synthesize

CONTACT = np.array([0.005, -0.003, -0.040])
HORIZON = 40
CLI_SAMPLES = 600
SHORT_SAMPLES = 250
# A vial too narrow for the stroke the tracker alone produces, so the constrained
# solve has to pull the blade in before it fits.
SAFE_VIAL = VialGeometry(body_radius=0.003, lip_radius=0.003)


@pytest.fixture(scope="module")
def generated(model, single_stroke):
    """A short generated stroke, orientation included."""
    return synthesize(model, vial_pose_of(single_stroke), CONTACT, horizon=HORIZON)


@pytest.fixture(scope="module")
def cli_dataset(tmp_path_factory):
    """Demonstrations on disk under a dataset name the calibration registry knows."""
    name = sorted(load_config()["datasets"])[0]
    return write_cli_dataset(tmp_path_factory.mktemp("cli") / name, CLI_SAMPLES)


@pytest.fixture(scope="module")
def short_cli_dataset(tmp_path_factory):
    """The same dataset over a shorter stroke, which the constrained solve is run on."""
    name = sorted(load_config()["datasets"])[0]
    return write_cli_dataset(tmp_path_factory.mktemp("cli_short") / name, SHORT_SAMPLES)


def test_saved_arrays_round_trip_bit_identical(tmp_path, generated):
    """Reading back a written stroke returns every array unchanged to the last bit."""
    path = save_trajectory(str(tmp_path / "stroke.npz"), generated)
    arrays, _ = load_trajectory(path)
    expected = {"time": generated.time, "position": generated.pos_world,
                "orientation": generated.quat_world, "velocity": generated.vel_world,
                "angular_velocity": generated.omega_world}
    assert set(arrays) == set(FIELDS)
    for name in FIELDS:
        assert_array_equal(arrays[name], expected[name])


def test_archive_loads_without_pickle_with_exactly_the_expected_keys(tmp_path, generated):
    """The archive is plain arrays: it opens with allow_pickle disabled."""
    path = save_trajectory(str(tmp_path / "stroke.npz"), generated)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(FIELDS) | {"metadata"}
        assert archive["time"].shape == (HORIZON,)
        assert archive["position"].shape == (HORIZON, 3)
        assert archive["orientation"].shape == (HORIZON, 4)
        assert archive["velocity"].shape == (HORIZON, 3)
        assert archive["angular_velocity"].shape == (HORIZON, 3)


def test_metadata_records_the_solve_and_its_units(tmp_path, generated):
    """The metadata states the timestep, the solve and the units the arrays are in."""
    path = save_trajectory(str(tmp_path / "stroke.npz"), generated)
    _, metadata = load_trajectory(path)
    assert metadata["dt"] == generated.dt == pytest.approx(0.02, rel=1e-12)
    assert metadata["constrained"] is False
    assert metadata["solver_status"] == ""
    assert metadata["jerk_rms"] == pytest.approx(generated.jerk_rms, rel=1e-12)
    assert metadata["tracking_rms"] == pytest.approx(generated.tracking_rms, rel=1e-12)
    assert metadata["frame"] == "world"
    assert metadata["quaternion_order"] == "xyzw"
    assert metadata["units"] == {"position": "m", "velocity": "m/s",
                                 "angular_velocity": "rad/s", "time": "s"}


def test_caller_metadata_is_kept_alongside_the_recorded_settings(tmp_path, generated):
    """Keys passed by the caller survive the round trip next to the generated ones."""
    path = save_trajectory(str(tmp_path / "stroke.npz"), generated, metadata={
        "states": 4, "contact_point_vial_frame": CONTACT.tolist()})
    _, metadata = load_trajectory(path)
    assert metadata["states"] == 4
    assert metadata["contact_point_vial_frame"] == CONTACT.tolist()
    assert metadata["frame"] == "world"


def test_saving_a_stroke_without_orientation_is_refused(tmp_path, position_model,
                                                        single_stroke):
    """A position-only stroke has no orientation to write, so writing it is an error."""
    flat = synthesize(position_model, vial_pose_of(single_stroke), CONTACT, horizon=HORIZON)
    assert flat.quat_world is None
    with pytest.raises(ValueError, match="orientation"):
        save_trajectory(str(tmp_path / "flat.npz"), flat)
    assert not os.path.exists(tmp_path / "flat.npz")


@pytest.mark.parametrize("given,written", [("stroke", "stroke.npz"),
                                           ("stroke.npz", "stroke.npz")])
def test_returned_path_names_the_file_on_disk(tmp_path, generated, given, written):
    """The returned path carries the suffix, whether or not the caller gave one."""
    returned = save_trajectory(str(tmp_path / given), generated)
    assert os.path.basename(returned) == written
    assert os.path.exists(returned)
    assert len(load_trajectory(returned)[0]["time"]) == HORIZON


def test_generate_flags_parse_into_typed_values():
    """The generate command turns its flags into numbers and a command to run."""
    args = cli.build_parser().parse_args(
        ["generate", "demos", "--states", "3", "--contact", "0.01", "-0.02", "-0.05",
         "--tilt", "60", "--contact-orientation", "radial", "--safe", "--anchor-end",
         "--out", "s.npz"])
    assert args.function is cli.generate
    assert (args.data_dir, args.states, args.out) == ("demos", 3, "s.npz")
    assert args.contact == [0.01, -0.02, -0.05]
    assert args.tilt == pytest.approx(60.0)
    assert args.contact_orientation == "radial"
    assert (args.safe, args.anchor_end, args.flat_ends) == (True, True, False)


def test_evaluate_parses_only_its_own_flags():
    """The evaluate command reads the shared flags and none of the generate ones."""
    args = cli.build_parser().parse_args(
        ["evaluate", "demos", "--states", "8", "--control-cost", "0.5",
         "--orientation-frame", "contact"])
    assert args.function is cli.evaluate
    assert (args.data_dir, args.states) == ("demos", 8)
    assert args.control_cost == pytest.approx(0.5)
    assert args.orientation_frame == "contact"
    assert not hasattr(args, "contact")


@pytest.mark.parametrize("argv", [["generate", "demos", "--contact-orientation", "spiral"],
                                  ["generate", "demos", "--orientation-frame", "bore"],
                                  ["demos"],
                                  []])
def test_an_unusable_command_line_exits_with_an_error(argv):
    """An unknown choice or a missing command exits with the usage error status."""
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(argv)
    assert exit_info.value.code == 2


def test_a_dataset_without_a_registered_calibration_is_refused(written_dataset):
    """A dataset the calibration registry does not know cannot be processed."""
    with pytest.raises(KeyError, match="no calibration"):
        cli.main(["generate", written_dataset, "--states", "3"])


def test_generate_writes_a_trajectory_as_long_as_the_model_horizon(cli_dataset, tmp_path,
                                                                   capsys):
    """The written stroke has one sample per timestep of the model's horizon."""
    out = tmp_path / "stroke.npz"
    cli.main(["generate", cli_dataset, "--states", "3", "--out", str(out)])
    printed = capsys.readouterr().out
    assert os.path.basename(cli_dataset) in printed
    assert str(out) in printed

    horizon = int(round(np.mean(
        [s.n_samples for s in prepare_dataset(load_dataset(cli_dataset))])))
    arrays, metadata = load_trajectory(str(out))
    assert arrays["time"].shape == (horizon,)
    assert arrays["position"].shape == (horizon, 3)
    assert metadata["dataset"] == os.path.basename(cli_dataset)
    assert metadata["states"] == 3
    assert metadata["contact_orientation"] == "vial"
    assert_allclose(np.linalg.norm(arrays["orientation"], axis=1), 1.0, atol=1e-9)
    assert_allclose(np.diff(arrays["time"]), metadata["dt"], rtol=1e-9)


def test_generate_under_the_constraints_writes_a_stroke_that_fits_the_vial(
        short_cli_dataset, tmp_path):
    """A constrained run records the solve and fits a vial the unconstrained stroke leaves."""
    settings = SafetySettings(vial=SAFE_VIAL)
    narrow = ["--vial-radius", str(SAFE_VIAL.body_radius),
              "--lip-radius", str(SAFE_VIAL.lip_radius)]
    free, safe = tmp_path / "free.npz", tmp_path / "safe.npz"
    cli.main(["generate", short_cli_dataset, "--states", "3", "--flat-ends",
              "--out", str(free)])
    cli.main(["generate", short_cli_dataset, "--states", "3", "--flat-ends", "--safe",
              *narrow, "--out", str(safe)])
    vial_frame = prepare_dataset(load_dataset(short_cli_dataset))[0].frames["vial"]

    arrays, metadata = load_trajectory(str(safe))
    assert metadata["constrained"] is True
    assert metadata["solver_status"].startswith("optimal")
    wall, lip = containment_margins(arrays["position"], arrays["orientation"],
                                    vial_frame, settings)
    assert np.isfinite(wall).sum() > 5 and np.isfinite(lip).sum() > 5
    assert min(np.nanmin(wall), np.nanmin(lip)) > -1e-4

    loose, loose_metadata = load_trajectory(str(free))
    assert (loose_metadata["constrained"], loose_metadata["solver_status"]) == (False, "")
    wall, lip = containment_margins(loose["position"], loose["orientation"],
                                    vial_frame, settings)
    assert min(np.nanmin(wall), np.nanmin(lip)) < -4e-4


def test_evaluate_reports_a_small_reproduction_error(cli_dataset, capsys):
    """Replaying the demonstrations reports a millimetre position and sub-degree angle."""
    cli.main(["evaluate", cli_dataset, "--states", "3"])
    printed = capsys.readouterr().out
    for name in sorted(os.listdir(cli_dataset)):
        assert name in printed

    mean = re.search(r"mean\s+([0-9.]+) mm\s+([0-9.]+) deg", printed)
    assert mean is not None, printed
    position_mm, orientation_deg = float(mean.group(1)), float(mean.group(2))
    assert 0.0 < position_mm < 2.0
    assert 0.0 < orientation_deg < 0.5
