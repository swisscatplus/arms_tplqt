"""Fixtures for the test suite.

The tests run on synthetic demonstrations built here, so the suite needs no
recorded data: a stroke is generated in the vial frame, given an orientation, and
pushed out into the motion-capture frame through the inverse of a calibration.
Everything downstream then runs through the real loading and preprocessing code.

The one fixture that does not build its data here is ``recorded_strokes``, which
prepares the demonstrations of a recorded dataset instead; the tests that ask for
it skip unless ``TPLQT_DATASET`` holds the path of one.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from tplqt.calibration import calibration_for_dataset
from tplqt.dataset import Demonstration, load_dataset
from tplqt.frames import aim_tool_axis, mocap_to_world, world_to_mocap
from tplqt.model import fit
from tplqt.preprocess import prepare_dataset

CALIBRATION = (np.array([-0.33, 0.40, 0.027]),
               R.from_rotvec([0.01, -0.02, 0.03]).as_quat())
RATE_HZ = 250.0
N_CLI_DEMOS = 3
VIAL_POSITION = np.array([0.30, -0.10, 0.20])
VIAL_TILT_DEG = 18.0


def vial_pose(seed: int = 0):
    """A vial pose in world: tilted, and rotated about its own bore by the seed."""
    tilt = R.from_rotvec(np.radians(VIAL_TILT_DEG) * np.array([1.0, 0.0, 0.0]))
    spin = R.from_rotvec(0.4 * seed * np.array([0.0, 0.0, 1.0]))
    return VIAL_POSITION + 0.01 * seed * np.array([1.0, -1.0, 0.0]), (tilt * spin).as_quat()


def scoop_path_vial(contact, n_samples: int = 600):
    """A stroke in the vial frame: in through the lip, down to ``contact``, back out."""
    phase = np.linspace(0.0, 1.0, n_samples)
    depth = np.where(phase < 0.5, phase / 0.5, (1.0 - phase) / 0.5)      # 0 -> 1 -> 0
    smooth_depth = 0.5 - 0.5 * np.cos(np.pi * depth)
    lateral = np.sin(np.pi * phase) ** 2
    path = np.zeros((n_samples, 3))
    path[:, 0] = contact[0] * lateral
    path[:, 1] = contact[1] * lateral
    # The stroke starts just above the lip and dips to the contact depth.
    path[:, 2] = 0.004 - (0.004 - contact[2]) * smooth_depth
    return path, int(np.argmax(smooth_depth))


def synthetic_demonstration(index: int = 0, *, contact=None, n_samples: int = 600
                            ) -> Demonstration:
    """One synthetic demonstration, recorded the way the real ones are."""
    rng = np.random.default_rng(index)
    if contact is None:
        contact = np.array([0.004, -0.003, -0.040]) + rng.normal(scale=0.0008, size=3)
    position, quaternion = vial_pose(index)
    A_vial = R.from_quat(quaternion).as_matrix()

    path_vial, contact_index = scoop_path_vial(contact, n_samples)
    pos_world = path_vial @ A_vial.T + position

    # The spatula points into the vial, leaning a little further over the stroke.
    drift = 0.02 * rng.normal(size=n_samples).cumsum() / n_samples
    lean = np.linspace(-0.12, 0.12, n_samples) + drift
    reference = aim_tool_axis(-A_vial[:, 2], np.array([0.0, 0.0, 0.0, 1.0]))
    quat_world = np.array([
        (R.from_rotvec(lean[t] * A_vial[:, 0]) * R.from_quat(reference)).as_quat()
        for t in range(n_samples)])

    pos_mocap, quat_mocap = world_to_mocap(pos_world, quat_world, calibration=CALIBRATION)
    time = np.arange(n_samples) / RATE_HZ + 1_700_000_000.0

    return Demonstration(
        name=f"synthetic_{index:02d}",
        dataset="synthetic",
        time=time,
        spatula_pos=pos_mocap,
        spatula_quat=quat_mocap,
        wrench=np.zeros((n_samples, 6)),
        vial_lip_pos=position,
        vial_lip_quat=quaternion,
        contact_time=float(time[contact_index]),
        contact_pos_mocap=pos_mocap[contact_index],
        contact_quat_mocap=quat_mocap[contact_index],
        sampled_mass_mg=5.0,
        descriptors={"primary": "synthetic"},
        rate_hz=RATE_HZ,
    )


def write_demonstration(folder: str, demo: Demonstration) -> str:
    """Write a demonstration to disk in the recorded format."""
    os.makedirs(folder, exist_ok=True)

    def stream(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def xyz(v):
        return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

    def xyzw(q):
        return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}

    stream(os.path.join(folder, "spatula_pose_in_mocap_frame.jsonl"),
           [{"time": float(t), "position": xyz(p), "orientation": xyzw(q)}
            for t, p, q in zip(demo.time, demo.spatula_pos, demo.spatula_quat)])
    stream(os.path.join(folder, "vial_lip_in_world.jsonl"),
           [{"time": float(t), "position": xyz(demo.vial_lip_pos),
             "orientation": xyzw(demo.vial_lip_quat)} for t in demo.time])
    stream(os.path.join(folder, "force_torque_data_in_bota_frame.jsonl"),
           [{"time": float(t), "force": xyz(w[:3]), "torque": xyz(w[3:])}
            for t, w in zip(demo.time, demo.wrench)])
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump({
            "demo_name": demo.name,
            "contact_time": demo.contact_time,
            "contact_pose": {"time": demo.contact_time,
                             "position": xyz(demo.contact_pos_mocap),
                             "orientation": xyzw(demo.contact_quat_mocap)},
            "scoop_weight_mg": demo.sampled_mass_mg,
            "descriptors": demo.descriptors,
            "recording_rate_hz": demo.rate_hz,
        }, f)
    return folder


def write_cli_dataset(root, n_samples):
    """Write demonstrations into ``root``, in the frame of the calibration it names.

    The command line looks a dataset's calibration up by the name of its
    directory, so a dataset written for it has to be named after one the
    registry knows and expressed in that calibration's frame.
    """
    calib = calibration_for_dataset(os.path.basename(root))
    for index in range(N_CLI_DEMOS):
        demo = synthetic_demonstration(index, n_samples=n_samples)
        pos, quat = mocap_to_world(demo.spatula_pos, demo.spatula_quat,
                                   calibration=CALIBRATION)
        demo.spatula_pos, demo.spatula_quat = world_to_mocap(pos, quat, calibration=calib)
        contact = int(np.argmin(np.abs(demo.time - demo.contact_time)))
        demo.contact_pos_mocap = demo.spatula_pos[contact]
        demo.contact_quat_mocap = demo.spatula_quat[contact]
        write_demonstration(str(root / demo.name), demo)
    return str(root)


@pytest.fixture(scope="session")
def demonstrations():
    """Six synthetic demonstrations, differing in vial pose and contact point."""
    return [synthetic_demonstration(i) for i in range(6)]


@pytest.fixture(scope="session")
def strokes(demonstrations):
    """The synthetic demonstrations prepared with a bore-aligned contact frame."""
    return prepare_dataset(demonstrations, calibration=CALIBRATION)


@pytest.fixture(scope="session")
def radial_strokes(demonstrations):
    """The same demonstrations prepared with a radial contact frame."""
    return prepare_dataset(demonstrations, calibration=CALIBRATION,
                           contact_orientation="radial")


@pytest.fixture(scope="session")
def model(strokes):
    """A model fitted on the synthetic demonstrations, orientation included."""
    return fit(strokes, n_states=4)


@pytest.fixture(scope="session")
def radial_model(radial_strokes):
    """A model whose chart is anchored in the radial contact frame."""
    return fit(radial_strokes, n_states=4)


@pytest.fixture(scope="session")
def position_model(strokes):
    """A position-only model."""
    return fit(strokes, n_states=4, orientation=False)


@pytest.fixture
def written_dataset(tmp_path, demonstrations):
    """The synthetic demonstrations written to disk as a dataset directory."""
    root = tmp_path / "synthetic"
    for demo in demonstrations:
        write_demonstration(str(root / demo.name), demo)
    return str(root)


@pytest.fixture(scope="session")
def recorded_strokes():
    """Strokes from a recorded dataset, skipped unless TPLQT_DATASET is set."""
    path = os.environ.get("TPLQT_DATASET")
    if not path:
        pytest.skip("set TPLQT_DATASET to a recorded dataset to run this test")
    return prepare_dataset(load_dataset(path))


@pytest.fixture(scope="session")
def single_stroke(strokes):
    return strokes[0]


def vial_pose_of(stroke):
    """``(position, quaternion)`` of the vial a stroke was demonstrated in."""
    A, b = stroke.frames["vial"]
    return b, R.from_matrix(A).as_quat()
