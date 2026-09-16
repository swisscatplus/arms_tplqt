"""Fit a model on a dataset and generate a stroke for a new contact point.

Run from the package root, so that the package is importable:

    python3 -m examples.generate_stroke data/salt_scoop
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

import tplqt

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/salt_scoop"

# Read the demonstrations and reduce each to the stroke inside the vial.
strokes = tplqt.prepare_dataset(tplqt.load_dataset(data_dir))

# Fit the task-parameterised model: six states, position and orientation.
model = tplqt.fit(strokes, n_states=6)

# Measure what the model kept, by replaying the demonstrations it was fitted on.
errors = [(tplqt.position_rms(s, r), tplqt.orientation_rms_deg(s, r))
          for s, r in ((s, tplqt.reproduce(model, s)) for s in strokes)]
position, orientation = np.mean(errors, axis=0)
print(f"reproduction error: {position * 1e3:.2f} mm, {orientation:.2f} deg")

# Generate a stroke for a vial pose and a contact point 5 mm off the vial axis,
# keeping the spatula inside the vial.
A_vial, b_vial = strokes[0].frames["vial"]
vial_pose = (b_vial, R.from_matrix(A_vial).as_quat())
contact_point = np.array([0.005, 0.0, -0.040])

stroke = tplqt.synthesize(model, vial_pose, contact_point, tilt=72.0,
                          safety=tplqt.SafetySettings())

target = A_vial @ contact_point + b_vial
reach = np.linalg.norm(stroke.pos_world - target, axis=1).min()
wall, lip = tplqt.containment_margins(stroke.pos_world, stroke.quat_world,
                                      (A_vial, b_vial), tplqt.SafetySettings())
print(f"generated {len(stroke.xi)} samples at {1 / stroke.dt:.0f} Hz")
print(f"closest approach to the contact point: {reach * 1e3:.2f} mm")
print(f"clearance from the wall: {np.nanmin(wall) * 1e3:.2f} mm")

tplqt.save_trajectory("stroke.npz", stroke, metadata={"dataset": data_dir})
print("wrote stroke.npz")
