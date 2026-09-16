# Task parameterised linear quadratic tracking

Task-parameterised learning and generation of spatula strokes.

A chemist sampling a solid moves a spatula into a vial, scrapes material from the
wall or the bottom and lifts it out. `tplqt` learns that stroke from
demonstrations and regenerates it for a vial that stands somewhere else and a
contact point that was never demonstrated, as a full Cartesian trajectory --
position, orientation, and the velocities to feed forward -- which it can be
required to keep inside the vial.

The model is a task-parameterised hidden Markov model: every demonstration is
described at once from the vial frame and from the contact frame, so the fitted
Gaussians record which parts of the stroke are organised by the vial and which by
where the material is touched. For a new situation the frames are rebuilt, each
frame's Gaussians are mapped into world through its own transform and multiplied
state by state, and the resulting sequence of Gaussians is tracked by a linear
quadratic tracker. Orientation is carried through the same machinery by
coordinatising it in a chart anchored to a task frame, which keeps the model
equivariant: move the vial and the whole stroke, spatula lean included, moves with
it.

The package depends only on numpy, scipy and cvxpy.

## Installation

The package needs numpy, scipy and cvxpy:

```bash
python3 -m pip install numpy scipy cvxpy
```

Nothing else has to be installed: every command below is run from this directory,
where this file is. Installing the package as well puts an equivalent `tplqt`
command on your path, so the same commands work anywhere without the `python3 -m`
prefix:

```bash
python3 -m pip install -e .
```

## Data

Three sets of demonstrations ship in `data/`, all recorded on the same workbench
with the same instrumented spatula:

| dataset | demonstrations | what was demonstrated |
| --- | --- | --- |
| `data/salt_scoop` | 19 | scooping salt, a grainy solid; contacts 1 to 11 mm from the bore axis, 32 to 52 mm deep |
| `data/honey_scoop` | 20 | scooping honey, viscous and cohesive; contacts on the axis, and longer strokes |
| `data/honey_deposit` | 23 | depositing honey against the wall; contacts 14 to 19 mm from the axis |

Any recording with the layout below works, and `tests/conftest.py` writes
synthetic demonstrations in exactly this format if you want a worked reference.

A dataset is a directory of demonstration folders, each holding four files:

```
data/salt_scoop/
  2026_06_15_14_05_10_salt_01/
    metadata.json                          name, contact instant and pose, sampled mass
    spatula_pose_in_mocap_frame.jsonl      spatula pose stream, motion-capture frame
    vial_lip_in_world.jsonl                vial lip pose stream, world frame
    force_torque_data_in_bota_frame.jsonl  wrench stream
  2026_06_15_14_06_12_salt_02/
  ...
```

Each line of a stream is a JSON object with a `time` in seconds and its payload:
`position` and `orientation` as `{x, y, z}` and `{x, y, z, w}` for the two pose
streams, `force` and `torque` as `{x, y, z}` for the wrench. `metadata.json` gives
the demonstration name, `contact_time`, the spatula `contact_pose` at that instant,
the sampled mass and the recording rate. The three streams share a clock and are
sampled row for row, and the vial neither moves nor turns within a recording; both
are checked at load time.

Spatula poses are recorded in the motion-capture frame and the vial in the world
frame of the workbench, so each dataset needs the hand-eye calibration that was in
force when it was recorded. `tplqt/calibrations.json`, inside the package, holds
those calibrations and the dataset each one applies to. For a dataset that is not listed, add a row there,
name an existing calibration with `python3 -m tplqt ... --calibration <date>`, pass
`calibration=(translation, quaternion)` to `prepare` directly, or read a file of
your own with `tplqt.calibration.load_config(path)`.

## Quick start

Fit a model and generate a stroke, as a scoop:

```bash
python3 -m tplqt generate data/salt_scoop --contact 0 0 -0.043 --safe --out salt_scoop.npz
python3 -m tplqt generate data/honey_scoop --contact 0 0 -0.043 --safe --out honey_scoop.npz
```

As a deposit against the vial wall, where the contact frame follows the azimuth of
the contact point and the spatula leans towards the wall it deposits on:

```bash
python3 -m tplqt generate data/honey_deposit --contact-orientation radial \
    --contact 0.0 0.011 -0.045 --flat-ends --out honey_deposit.npz
```

The honey deposit contacts were recorded 14 to 19 mm from the bore axis and 37 to 57 mm
below the lip, so a contact outside that range is an extrapolation and the printed
distance to the demonstrations says so.

Replay every demonstration with the fitted model to see what it kept:

```bash
python3 -m tplqt evaluate data/salt_scoop
python3 -m tplqt evaluate data/honey_scoop
python3 -m tplqt evaluate data/honey_deposit --contact-orientation radial
```

Averaged over the demonstrations, these report a reproduction error of 2.8 mm and
2.4 degrees, 3.7 mm and 3.1 degrees, and 3.4 mm and 4.9 degrees respectively,
against demonstrations that themselves spread 5 to 12 mm within a state.

The generated stroke is written as a `.npz` of plain arrays -- `time`,
`position`, `orientation` as xyzw quaternions, `velocity`, `angular_velocity`, all
in world coordinates and SI units, plus a JSON `metadata` string carrying the vial
pose and contact point the stroke was generated for. It loads without
`allow_pickle`:

```python
import tplqt
arrays, metadata = tplqt.load_trajectory("salt_scoop.npz")
```

## Using it as a library

```python
import numpy as np
import tplqt
from scipy.spatial.transform import Rotation as R

strokes = tplqt.prepare_dataset(tplqt.load_dataset("data/salt_scoop"))
model = tplqt.fit(strokes, n_states=6)

A_vial, b_vial = strokes[0].frames["vial"]
stroke = tplqt.synthesize(model,
                          vial_pose=(b_vial, R.from_matrix(A_vial).as_quat()),
                          contact_point=np.array([0.005, 0.0, -0.040]),
                          tilt=72.0,
                          safety=tplqt.SafetySettings())
```

`examples/generate_stroke.py` is this, end to end:

```bash
python3 -m examples.generate_stroke data/salt_scoop
```

## How the pieces fit

| module | what it does |
| --- | --- |
| `dataset` | read recorded demonstrations |
| `calibration` | the motion-capture to world calibration of each dataset |
| `frames` | coordinate transforms, the vial and contact task frames, spatula aiming |
| `preprocess` | window each recording to the stroke inside the vial, smooth, downsample, differentiate |
| `orientation` | the orientation chart, its Jacobians, and the angles used to report error |
| `gaussian` | Gaussian mixtures: marginal, affine transform, product of Gaussians |
| `hmm` | hidden Markov model with Gaussian emissions, fitted by Baum-Welch |
| `lqt` | the chain of integrators and the batch linear quadratic tracker |
| `model` | the task-parameterised model: per-frame observations, fitting, state schedule |
| `reproduce` | replay a demonstration and measure the error |
| `synthesize` | generate a stroke for a new vial pose and contact point |
| `safety` | re-solve that stroke so the spatula stays inside the vial |
| `export` | write and read a generated stroke |
| `cli` | the `generate` and `evaluate` commands |

### The two task frames

The vial frame sits at the centre of the lip with its z axis along the bore,
pointing out of the opening. The contact frame sits at the contact point; its
orientation is either the vial's, which makes it a pure translation and suits a
scoop organised by depth, or radial, turned about the bore by the contact's
azimuth so that its x axis is the inward wall normal. A stroke learned in a radial
contact frame generalises around the wall instead of being tied to one side of the
vial, which is what depositing against the wall needs.

### Orientation

Orientations are coordinatised by their deviation from a reference,
`eta = Log(q_ref^-1 q)`, which gives the Gaussian model and the tracker three more
configuration coordinates to work with alongside position. The reference is stored
relative to a task frame -- the vial frame, or the contact frame when the contact
frame is radial -- and rotated back with that frame when a stroke is generated, so
the model is equivariant under moving the vial. Anchoring the chart in the world
instead is the naive alternative the frame-anchored one is compared against
(`orientation_frame="world"`); it pins the nominal orientation where the
demonstrations happened to put it, and is not equivariant.

Equivariance is a property of the model, not of every stroke it generates: a start
pose asked for by its tilt from gravity is referenced to a world direction, so a
stroke started that way follows the vial only up to rotations about the vertical.
Pass `tilt=None`, or an explicit start state, to keep the whole stroke equivariant.

`chart_excursion_deg` reports how far a dataset strays from the reference, which is
the check on the small-angle assumption the linearisation rests on.

### Staying inside the vial

The tracker knows nothing about the vial. `SafetySettings` turns the stroke into a
second-order cone program that keeps every sampled point of the blade within the
wall at its own depth and puts the crossing of the lip plane inside the mouth,
under the same tracking cost. The spatula's direction depends on its orientation,
which is linearised about the current stroke and re-solved a few times, so the
solver can tilt the spatula to clear the wall instead of only translating it.

The constraints are imposed on that linearisation, so the stroke that comes out is
checked against the true geometry before it is returned, and an infeasible problem
or a stroke that fails the check raises `InfeasibleTrajectory` rather than
returning something that would hit the vial. `containment_margins` and
`worst_violation` are the same check, to run on any stroke.

The vial is described by `VialGeometry` and the blade by `SpatulaGeometry`, both in
metres; the defaults are the vial and the spatula the scooping demonstrations were
recorded with. A contact point further from the bore axis than the wall at its depth cannot
be reached by a stroke that stays inside: the constrained stroke stops at the wall,
and a warning says by how much the contact is out of reach. The deposit
demonstrations are recorded against the wall, 14 to 19 mm from the bore axis, so
generating a deposit under the scooping vial's 12.5 mm radius does exactly that --
give `VialGeometry` the geometry the deposit was recorded in, or generate it
without the constraints.

## Tests

```bash
python3 -m pip install pytest
python3 -m pytest                                    # the suite, no data needed
TPLQT_DATASET=data/salt_scoop python3 -m pytest       # also the tests that use a dataset
```

## Citation

If you use this code, please cite the accompanying paper. The full reference is
added here on acceptance.

## License

MIT, see `LICENSE`.
