"""Command line for fitting a model, generating a stroke and looking at it.

Three commands:

``generate``
    fit a model on a dataset, generate a stroke for a vial pose and a contact
    point, report how it tracks the demonstrations and optionally write it to a
    ``.npz``;
``evaluate``
    fit a model and replay every demonstration, reporting the reproduction error;
``view``
    draw generated strokes in the vial they were generated for, as a rerun
    recording or a video.

Run them as ``python -m tplqt <command>``.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .calibration import calibration, calibration_names
from .dataset import dataset_name, load_dataset
from .export import load_trajectory, save_trajectory
from .frames import CONTACT_ORIENTATIONS, aim_tool_axis
from .model import ORIENTATION_FRAMES, fit
from .preprocess import prepare_dataset
from .reproduce import orientation_rms_deg, position_rms, reproduce
from .safety import SafetySettings, SpatulaGeometry, VialGeometry, containment_margins
from .synthesize import path_rms_by_state, synthesize
from .viz import Scene, Trajectory, entity_name, record_rrd, show


def _fit(args):
    """Load, prepare and fit, echoing the choices that shape the model."""
    chosen = calibration(args.calibration) if args.calibration else None
    strokes = prepare_dataset(load_dataset(args.data_dir), calibration=chosen,
                              contact_orientation=args.contact_orientation)
    model = fit(strokes, n_states=args.states, orientation_frame=args.orientation_frame)
    print(f"{dataset_name(args.data_dir)}: {len(strokes)} demonstrations, "
          f"{args.states} states, contact frame {model.contact_orientation}, "
          f"orientation chart anchored in the {model.orientation_frame} frame")
    print(f"  {model.rate_hz:.1f} Hz, average stroke {model.horizon} samples")
    return strokes, model


def _safety_settings(args) -> Optional[SafetySettings]:
    if not args.safe:
        return None
    return SafetySettings(
        vial=VialGeometry(body_radius=args.vial_radius, lip_radius=args.lip_radius),
        spatula=SpatulaGeometry(length=args.blade_length),
        margin=args.margin,
    )


def generate(args):
    strokes, model = _fit(args)
    settings = _safety_settings(args)
    anchor = strokes[args.demo]
    A_vial, b_vial = anchor.frames["vial"]
    vial_pose = (b_vial, R.from_matrix(A_vial).as_quat())
    contact = np.asarray(args.contact, float)

    boundary = None
    if args.flat_ends:
        boundary = aim_tool_axis(-A_vial[:, 2], anchor.quat_world[0])

    stroke = synthesize(model, vial_pose, contact, control_cost=args.control_cost,
                        tilt=args.tilt, start_shift=args.start_shift,
                        boundary_orientation=boundary, anchor_end=args.anchor_end,
                        safety=settings)

    target = A_vial @ contact + b_vial
    reach = float(np.linalg.norm(stroke.pos_world - target, axis=1).min())
    start = ("with the spatula along the bore" if boundary is not None
             else f"at {args.tilt} deg from gravity")
    print(f"  contact point {np.round(contact, 4).tolist()} in the vial frame, "
          f"start at the lip {np.round(args.start_shift, 4).tolist()} {start}")
    print(f"  closest approach to the contact point {reach * 1e3:.2f} mm, "
          f"depth reached {stroke.pos_vial[:, 2].min() * 1e3:.1f} mm, "
          f"jerk {stroke.jerk_rms:.3f} m/s^3")

    rows, overall = path_rms_by_state(model, strokes, stroke)
    print(f"  {'state':>5} {'samples':>8} {'distance to demonstrations':>28} "
          f"{'their spread':>14}")
    for row in rows:
        print(f"  {row['state']:>5d} {row['n_generated']:>8d} "
              f"{row['rms'] * 1e3:>25.2f} mm {row['demonstration_spread'] * 1e3:>11.2f} mm")
    print(f"  overall distance to the demonstrations {overall * 1e3:.2f} mm")

    if settings is not None:
        wall, lip = containment_margins(stroke.pos_world, stroke.quat_world,
                                        (A_vial, b_vial), settings)
        print(f"  solver {stroke.solver_status}; clearance from the wall "
              f"{np.nanmin(wall) * 1e3:+.3f} mm, from the mouth "
              f"{np.nanmin(lip) * 1e3:+.3f} mm")

    if args.out:
        path = save_trajectory(args.out, stroke, metadata={
            "dataset": dataset_name(args.data_dir),
            "states": args.states,
            "vial_position": vial_pose[0].tolist(),
            "vial_quaternion": vial_pose[1].tolist(),
            "contact_point_vial_frame": contact.tolist(),
            "contact_orientation": model.contact_orientation,
            "orientation_frame": model.orientation_frame,
        })
        print(f"  wrote {path}")
        print(f"  look at it with: python3 -m tplqt view {path} "
              f"--demos {args.data_dir}")


def evaluate(args):
    strokes, model = _fit(args)
    print(f"  {'demonstration':<34} {'position':>10} {'orientation':>13}")
    position, orientation = [], []
    for stroke in strokes:
        replay = reproduce(model, stroke, control_cost=args.control_cost)
        position.append(position_rms(stroke, replay))
        orientation.append(orientation_rms_deg(stroke, replay))
        print(f"  {stroke.name:<34} {position[-1] * 1e3:>7.2f} mm "
              f"{orientation[-1]:>10.2f} deg")
    print(f"  {'mean':<34} {np.mean(position) * 1e3:>7.2f} mm "
          f"{np.mean(orientation):>10.2f} deg")
    print(f"  {'standard deviation':<34} {np.std(position) * 1e3:>7.2f} mm "
          f"{np.std(orientation):>10.2f} deg")


# Two trajectories may take the same vial pose from different demonstrations of it,
# so a scene they can share is one they agree on at the scale it is drawn at.
SCENE_TOLERANCE_M = 1e-3
SCENE_TOLERANCE_DEG = 1.0


def _scene_difference(first: dict, other: dict) -> Optional[str]:
    """How two trajectories disagree about the scene, or ``None`` if they do not.

    Everything the scene is drawn and measured in -- the vial pose and the contact
    point -- is taken from the first trajectory, so a second one generated for a
    different vial would be drawn outside it and measured against the wrong frame.
    """
    def moved(key, what):
        if (key in first) != (key in other):
            return f"one of them records no {what}"
        if key in first:
            distance = float(np.linalg.norm(np.asarray(first[key], float)
                                            - np.asarray(other[key], float)))
            if distance > SCENE_TOLERANCE_M:
                return f"their {what}s are {distance * 1e3:.1f} mm apart"
        return None

    for key, what in (("vial_position", "vial"), ("contact_point_vial_frame",
                                                  "contact point")):
        difference = moved(key, what)
        if difference is not None:
            return difference
    if ("vial_quaternion" in first) != ("vial_quaternion" in other):
        return "one of them records no vial orientation"
    if "vial_quaternion" in first:
        turned = np.degrees((R.from_quat(first["vial_quaternion"]).inv()
                             * R.from_quat(other["vial_quaternion"])).magnitude())
        if turned > SCENE_TOLERANCE_DEG:
            return f"their vials are turned {turned:.1f} degrees from each other"
    return None


def _load_for_view(paths: List[str]):
    """Read the trajectories to draw, reporting what each one does."""
    trajectories, first, names = [], None, set()
    for path in paths:
        arrays, metadata = load_trajectory(path)
        if first is None:
            first = metadata
        elif _scene_difference(first, metadata) is not None:
            raise SystemExit(
                f"{path} was not generated for the same situation as {paths[0]}: "
                f"{_scene_difference(first, metadata)}. Only strokes of one vial and "
                f"one contact point can be drawn together, since the scene is built "
                f"and measured in that one frame; view them one at a time instead")
        name = os.path.splitext(os.path.basename(path))[0]
        while entity_name(name) in names:           # two files may share a basename
            name += "_"
        names.add(entity_name(name))
        trajectories.append(Trajectory(arrays["position"], arrays["orientation"],
                                       arrays["velocity"], name=name))
        print(f"{path}: {len(arrays['time'])} samples of "
              f"{metadata.get('dataset', 'an unnamed dataset')}, "
              f"{'constrained' if metadata.get('constrained') else 'unconstrained'}, "
              f"tip speed up to "
              f"{np.linalg.norm(arrays['velocity'], axis=1).max() * 1e2:.1f} cm/s")
    return trajectories, first


def _vial_pose_for_view(args, metadata):
    """The vial pose to draw the scene in, from the command line or the file."""
    if args.vial_pose:
        return np.asarray(args.vial_pose[:3], float), np.asarray(args.vial_pose[3:], float)
    if "vial_position" not in metadata or "vial_quaternion" not in metadata:
        raise SystemExit(
            "this trajectory does not record the vial pose it was generated for, so "
            "there is nowhere to put the vial: give the pose with --vial-pose")
    return (np.asarray(metadata["vial_position"], float),
            np.asarray(metadata["vial_quaternion"], float))


def _clearance(trajectory, vial_frame, settings: SafetySettings) -> float:
    """Smallest clearance the blade keeps from the wall and the mouth, in metres.

    Negative means the blade reaches through the vial somewhere along the stroke,
    which is what an unconstrained stroke near the wall does.
    """
    wall, lip = containment_margins(trajectory.position, trajectory.orientation,
                                    vial_frame, settings)
    crossings = lip[np.isfinite(lip)]
    return float(min(np.min(wall), np.min(crossings) if len(crossings) else np.inf))


def view(args):
    trajectories, metadata = _load_for_view(args.trajectories)
    vial_pose = _vial_pose_for_view(args, metadata)
    contact = metadata.get("contact_point_vial_frame")

    demonstrations = None
    if args.demos:
        strokes = prepare_dataset(
            load_dataset(args.demos),
            calibration=calibration(args.calibration) if args.calibration else None,
            contact_orientation=metadata.get("contact_orientation", "vial"))
        demonstrations = [stroke.pos_vial for stroke in strokes]
        print(f"  {len(strokes)} demonstrations from {args.demos} behind them")

    scene = Scene(vial_pose=vial_pose, trajectories=trajectories,
                  vial=VialGeometry(body_radius=args.vial_radius,
                                    lip_radius=args.lip_radius),
                  spatula=SpatulaGeometry(length=args.blade_length),
                  demonstrations=demonstrations, contact_point=contact,
                  dt=metadata.get("dt"),
                  name=metadata.get("dataset", trajectories[0].name))

    A_vial, b_vial = scene.frame
    target = None if contact is None else scene.to_world(contact)
    settings = SafetySettings(vial=scene.vial, spatula=scene.spatula)
    print(f"  {'stroke':<24} {'depth':>9} {'to the contact':>17} {'clearance':>12}")
    for trajectory in trajectories:
        depth = ((trajectory.position - b_vial) @ A_vial[:, 2]).min()
        reach = ("" if target is None else
                 f"{np.linalg.norm(trajectory.position - target, axis=1).min() * 1e3:.2f} mm")
        clearance = _clearance(trajectory, (A_vial, b_vial), settings)
        print(f"  {trajectory.name:<24} {depth * 1e3:>6.1f} mm {reach:>17} "
              f"{clearance * 1e3:>+9.3f} mm")

    if args.spawn:
        show(scene)
        print("  opened the rerun viewer")
    out = args.out
    if out is None and not args.spawn:
        out = os.path.splitext(args.trajectories[0])[0] + ".rrd"
    if out is not None:
        record_rrd(out, scene)
        print(f"  wrote {out}; open it with: rerun {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tplqt", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub, control_cost):
        sub.add_argument("data_dir", help="directory of demonstration folders")
        sub.add_argument("--states", type=int, default=6,
                         help="number of states of the hidden Markov model (default 6)")
        sub.add_argument("--contact-orientation", default="vial",
                         choices=list(CONTACT_ORIENTATIONS),
                         help="orientation of the contact frame: 'vial' shares the vial's "
                              "orientation, 'radial' points at the vial axis (default vial)")
        sub.add_argument("--orientation-frame", default=None, choices=list(ORIENTATION_FRAMES),
                         help="frame the orientation chart is anchored in (default: the "
                              "contact frame when it is radial, the vial frame otherwise)")
        sub.add_argument("--control-cost", type=float, default=control_cost,
                         help=f"weight on control effort; higher is smoother and looser "
                              f"(default {control_cost})")
        sub.add_argument("--calibration", default=None, choices=list(calibration_names()),
                         help="motion-capture calibration to lift the recordings with "
                              "(default: the one registered for the dataset directory)")

    generate_parser = commands.add_parser(
        "generate", help="generate a stroke for a vial pose and a contact point")
    common(generate_parser, 1.0)
    generate_parser.add_argument("--contact", type=float, nargs=3, default=[0.0, 0.0, -0.043],
                                 metavar=("X", "Y", "Z"),
                                 help="contact point in the vial frame, measured from the "
                                      "centre of the lip, in metres (default 0 0 -0.043)")
    generate_parser.add_argument("--tilt", type=float, default=72.0,
                                 help="angle of the spatula from gravity at the start, in "
                                      "degrees; 90 points it horizontally into the vial, and "
                                      "the default 72 suits the scooping demonstrations")
    generate_parser.add_argument("--start-shift", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                                 metavar=("X", "Y", "Z"),
                                 help="displacement of the starting tip position from the "
                                      "centre of the lip, in the vial frame")
    generate_parser.add_argument("--demo", type=int, default=0,
                                 help="demonstration whose vial pose the stroke is generated "
                                      "for (default 0)")
    generate_parser.add_argument("--anchor-end", action="store_true",
                                 help="require the stroke to end at the centre of the lip")
    generate_parser.add_argument("--flat-ends", action="store_true",
                                 help="hold the spatula along the bore at both ends, so the "
                                      "model shapes only the middle of the stroke")
    generate_parser.add_argument("--safe", action="store_true",
                                 help="keep the spatula inside the vial by solving the "
                                      "constrained problem")
    generate_parser.add_argument("--vial-radius", type=float,
                                 default=VialGeometry().body_radius,
                                 help="inner radius of the vial body, in metres")
    generate_parser.add_argument("--lip-radius", type=float, default=VialGeometry().lip_radius,
                                 help="radius of the vial opening, in metres")
    generate_parser.add_argument("--blade-length", type=float,
                                 default=SpatulaGeometry().length,
                                 help="length of the spatula blade, in metres")
    generate_parser.add_argument("--margin", type=float, default=0.0,
                                 help="clearance kept from the vial wall, in metres")
    generate_parser.add_argument("--out", help="write the stroke to this .npz")
    generate_parser.set_defaults(function=generate)

    evaluate_parser = commands.add_parser(
        "evaluate", help="replay every demonstration and report the reproduction error")
    common(evaluate_parser, 1e-4)
    evaluate_parser.set_defaults(function=evaluate)

    view_parser = commands.add_parser(
        "view", help="draw generated strokes in the vial they were generated for")
    view_parser.add_argument("trajectories", nargs="+",
                             help=".npz files written by generate --out; several are "
                                  "drawn together, each in its own colour")
    view_parser.add_argument("--demos", default=None,
                             help="dataset directory whose demonstrations are drawn "
                                  "behind the strokes for context")
    view_parser.add_argument("--calibration", default=None,
                             choices=list(calibration_names()),
                             help="motion-capture calibration to lift those "
                                  "demonstrations with (default: the one registered "
                                  "for the dataset directory)")
    view_parser.add_argument("--out", default=None,
                             help="write the recording here (default: the first "
                                  "trajectory with a .rrd suffix)")
    view_parser.add_argument("--spawn", action="store_true",
                             help="open the rerun viewer; nothing is written unless "
                                  "--out asks for it as well")
    view_parser.add_argument("--vial-pose", type=float, nargs=7,
                             metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                             help="pose of the centre of the lip in world, for a "
                                  "trajectory that does not record one")
    view_parser.add_argument("--vial-radius", type=float,
                             default=VialGeometry().body_radius,
                             help="inner radius of the vial body, in metres")
    view_parser.add_argument("--lip-radius", type=float, default=VialGeometry().lip_radius,
                             help="radius of the vial opening, in metres")
    view_parser.add_argument("--blade-length", type=float, default=SpatulaGeometry().length,
                             help="length of the spatula blade, in metres")
    view_parser.set_defaults(function=view)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
