"""Task-parameterised learning and generation of spatula strokes.

The pipeline, in the order it runs:

``dataset``
    read recorded demonstrations;
``calibration``, ``frames``
    lift them into the world frame and build the vial and contact task frames;
``preprocess``
    window each recording to the stroke inside the vial, smooth, downsample and
    differentiate it;
``orientation``
    coordinatise the spatula orientation as a vector in a chart anchored to a task
    frame;
``model``
    fit a task-parameterised hidden Markov model over the stacked per-frame views;
``reproduce``
    replay a demonstration to measure what the model kept;
``synthesize``
    generate a stroke for a new vial pose and contact point;
``safety``
    re-solve that stroke so the spatula stays inside the vial;
``viz``
    draw the stroke in the vial it was generated for, to look at.

``gaussian``, ``hmm`` and ``lqt`` hold the estimation and control machinery the
model is built on.
"""
from __future__ import annotations

from .calibration import calibration, calibration_for_dataset
from .dataset import Demonstration, dataset_name, load_dataset, load_demonstration
from .export import load_trajectory, save_trajectory
from .model import TaskParameterizedModel, fit
from .orientation import chart_excursion_deg, geodesic_angle_deg
from .preprocess import Stroke, prepare, prepare_dataset
from .reproduce import Reproduction, orientation_rms_deg, position_rms, reproduce
from .safety import (InfeasibleTrajectory, SafetySettings, SpatulaGeometry, VialGeometry,
                     containment_margins, worst_violation)
from .synthesize import Synthesis, mean_contact, path_rms_by_state, synthesize, task_frames
from .viz import Scene, Trajectory, record_rrd, show

__all__ = [
    "Demonstration", "Reproduction", "SafetySettings", "Scene", "SpatulaGeometry",
    "Stroke", "Synthesis", "TaskParameterizedModel", "Trajectory", "VialGeometry",
    "InfeasibleTrajectory",
    "calibration", "calibration_for_dataset", "chart_excursion_deg",
    "containment_margins", "dataset_name", "geodesic_angle_deg",
    "fit", "load_dataset", "load_demonstration", "load_trajectory", "mean_contact",
    "orientation_rms_deg", "path_rms_by_state", "position_rms", "prepare",
    "prepare_dataset", "record_rrd", "reproduce", "save_trajectory",
    "show", "synthesize", "task_frames", "worst_violation",
]

__version__ = "1.0.0"
