"""Keeping the generated stroke inside the vial.

The tracker of :mod:`tplqt.lqt` knows nothing about the vial: it follows the
Gaussians, and a stroke generated for a contact point near the wall can put the
spatula through it. This module re-solves the same tracking problem as a
second-order cone program with two geometric requirements added:

containment
    every sampled point of the blade that is below the opening stays within the
    vial wall at its own depth, ``||P(p - b)|| <= r(depth)``, where ``P`` projects
    onto the plane normal to the bore;
incidence
    when the blade straddles the opening, the point where its axis crosses the
    lip plane lies within the mouth, ``||P(p_lip - b)|| <= r_lip``. This is the
    condition for the spatula line to puncture the disc of the mouth.

Both are norms of affine functions of the decision variables, so the problem stays
convex. The blade's direction depends on the spatula's orientation, which enters
the state through the chart of :mod:`tplqt.orientation`; it is linearised around
the current trajectory and the solve is repeated a few times, so the solver can
tilt the spatula to satisfy the wall instead of only translating it. Which blade
points are below the opening, where the blade crosses it, and how wide the vial is
at each of them are all read off that same linearisation, so the stroke that comes
out is checked against the true geometry before it is returned
(:func:`worst_violation`), and rejected if it is outside.

The geometry the constraints use is the caller's: :class:`VialGeometry` describes
the vial the scooping demonstrations were recorded in, and a contact point further from the
bore axis than the wall at its depth cannot be reached by a stroke that stays
inside, which :func:`tplqt.synthesize.synthesize` warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from . import orientation as ori
from .frames import tool_axis_world


# Ridge added to the reference covariance before it is inverted into a cost weight.
WEIGHT_RIDGE = 1e-9

# A returned trajectory this far outside the vial is not a solution that missed its
# tolerance; it is what the solver hands back when the problem has none.
UNUSABLE_SOLUTION_M = 0.01


class InfeasibleTrajectory(RuntimeError):
    """No stroke was produced that keeps the spatula inside the vial.

    Raised either because the cone program has no solution, or because the stroke
    it returned does not survive the check against the true geometry. A stroke that
    cannot be made safe must not be executed, so this is raised rather than falling
    back to the unconstrained one. It carries the solver status and the iteration it
    failed at.
    """

    def __init__(self, status: str, iteration: Optional[int] = None):
        self.status = status
        self.iteration = iteration
        message = f"no stroke keeps the spatula inside the vial: {status}"
        if iteration is not None:
            message += f" (iteration {iteration})"
        super().__init__(message)


@dataclass(frozen=True)
class VialGeometry:
    """Inner shape of the vial, in metres.

    The vial is a cylinder of radius ``body_radius`` that narrows to
    ``lip_radius`` at the opening over a conical shoulder of length
    ``shoulder_length``, and is ``length`` deep. Depth is measured in the vial
    frame, zero at the lip and negative inside. The constraints are set by the
    radii and the shoulder; ``length`` says where the bottom is, which only
    :mod:`tplqt.viz` needs, to draw it.
    """

    body_radius: float = 0.01250
    lip_radius: float = 0.00850
    shoulder_length: float = 0.01480
    length: float = 0.05421


@dataclass(frozen=True)
class SpatulaGeometry:
    """The blade, modelled as a straight rod, in metres.

    ``length`` runs from the tip back along the blade, against the tool axis, and
    ``radius`` is half its thickness; ``n_samples`` points spread along it are
    constrained, from the tip to the base.
    """

    length: float = 0.10
    radius: float = 0.001
    n_samples: int = 4


@dataclass(frozen=True)
class SafetySettings:
    """How the geometric constraints are set up and solved.

    ``margin`` shrinks every radius, buying clearance from the wall.
    ``orientation_weight`` scales the cost of moving the orientation away from the
    tracked reference: below one the solver prefers tilting the spatula, above one
    it prefers translating it. It scales the orientation itself, not its rate, since
    that is what the constraints act on. ``max_iterations`` bounds the relinearisations of
    the spatula axis, which stop early once the orientation changes by less than
    ``tolerance_rad``. Because the constraints are imposed on a linearisation, the
    solved stroke is checked against the true geometry afterwards and rejected if it
    is outside by more than ``containment_tolerance`` metres.
    """

    vial: VialGeometry = field(default_factory=VialGeometry)
    spatula: SpatulaGeometry = field(default_factory=SpatulaGeometry)
    margin: float = 0.0
    orientation_weight: float = 1.0
    max_iterations: int = 3
    tolerance_rad: float = 1e-4
    containment_tolerance: float = 1e-4
    solver: str = "CLARABEL"
    verbose: bool = False


def wall_radius(depth, vial: VialGeometry = VialGeometry()) -> np.ndarray:
    """Radius of the vial wall at a given depth, elementwise.

    Depth is zero at the lip and negative inside: the radius is ``body_radius``
    below the shoulder, tapers linearly to ``lip_radius`` at the opening, and is
    unbounded above it, where the spatula is in free space.
    """
    depth = np.asarray(depth, float)
    taper = vial.lip_radius + (vial.body_radius - vial.lip_radius) * (
        -depth / vial.shoulder_length)
    return np.where(depth >= 0, np.inf,
                    np.where(depth <= -vial.shoulder_length, vial.body_radius, taper))


def blade_points(tip_world, tool_axis, length: float, samples) -> np.ndarray:
    """Points along the blade: ``tip + s * length * (-tool_axis)`` for each sample."""
    tip_world = np.atleast_2d(np.asarray(tip_world, float))
    tool_axis = np.atleast_2d(np.asarray(tool_axis, float))
    samples = np.asarray(samples, float)
    return tip_world[:, None, :] + samples[None, :, None] * length * (-tool_axis)[:, None, :]


def lip_crossing(tip_world, tool_axis, vial_frame, length: float):
    """Where the spatula axis crosses the plane of the vial opening.

    Returns ``(active, fraction, crossing_radius)``: whether the blade straddles
    the opening at each timestep, where along the blade it crosses, and how far
    from the bore axis the crossing sits. The blade straddles the opening exactly
    when its tip and its base are on opposite sides of the lip plane, which is
    tested as a product of depths and therefore stays well defined when the
    spatula lies nearly parallel to the plane.
    """
    A_vial, b_vial = vial_frame
    bore = A_vial[:, 2]
    cross_section = A_vial[:, 0:2].T
    tip_world = np.atleast_2d(np.asarray(tip_world, float))
    tool_axis = np.atleast_2d(np.asarray(tool_axis, float))

    tip_depth = (tip_world - b_vial) @ bore
    base_depth = tip_depth - length * (tool_axis @ bore)
    active = (tip_depth * base_depth) < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(active, tip_depth / (tip_depth - base_depth), np.nan)

    crossing = np.full(len(tip_world), np.nan)
    if active.any():
        point = (tip_world[active]
                 + (fraction[active] * length)[:, None] * (-tool_axis[active]))
        crossing[active] = np.linalg.norm((point - b_vial) @ cross_section.T, axis=1)
    return active, fraction, crossing


def containment_margins(tip_world, quat_world, vial_frame, settings: SafetySettings):
    """Clearance of a trajectory from the wall and from the mouth, in metres.

    Returns ``(wall, lip)``: the smallest clearance of any blade point below the
    opening at each timestep, and the clearance of the lip crossing where the
    blade straddles the opening (``nan`` where it does not). Negative means the
    constraint is violated. This is the independent check on a solved trajectory.
    """
    A_vial, b_vial = vial_frame
    bore = A_vial[:, 2]
    cross_section = A_vial[:, 0:2].T
    spatula = settings.spatula

    tool_axis = np.atleast_2d(tool_axis_world(quat_world))
    samples = np.linspace(0.0, 1.0, spatula.n_samples)
    points = blade_points(tip_world, tool_axis, spatula.length, samples)

    depth = (points - b_vial) @ bore
    radius = np.linalg.norm((points - b_vial) @ cross_section.T, axis=2)
    allowed = (wall_radius(depth, settings.vial) - spatula.radius - settings.margin)
    wall = np.where(depth < 0, allowed - radius, np.inf).min(axis=1)

    active, _, crossing = lip_crossing(tip_world, tool_axis, vial_frame, spatula.length)
    lip_allowed = settings.vial.lip_radius - spatula.radius - settings.margin
    lip = np.where(active, lip_allowed - crossing, np.nan)
    return wall, lip


def _solve_socp(*, A_dyn, B_dyn, x0, mu_seq, sigma_seq, control_cost,
                vial_frame, tip_linear, tool_axis_linear, eta_linear, tool_jacobian,
                settings: SafetySettings, anchor_end_steps: int, anchor_target):
    """One convex solve with the spatula axis linearised around a trajectory."""
    A_vial, b_vial = vial_frame
    bore = A_vial[:, 2]
    cross_section = A_vial[:, 0:2].T
    spatula = settings.spatula

    horizon, n_state = mu_seq.shape
    n_ctrl = np.asarray(B_dyn).shape[1]

    # The reference covariance is inverted with a small ridge: the boundary timesteps
    # of a pinned reference are deliberately near singular, and the cone solver needs a
    # finite weight there.
    weight = np.linalg.inv(sigma_seq + WEIGHT_RIDGE * np.eye(n_state))
    weight = 0.5 * (weight + np.swapaxes(weight, 1, 2))
    if settings.orientation_weight != 1.0:
        scale = np.eye(n_state)
        scale[3:6, 3:6] *= np.sqrt(settings.orientation_weight)
        weight = scale @ weight @ scale
        weight = 0.5 * (weight + np.swapaxes(weight, 1, 2))

    samples = np.linspace(0.0, 1.0, spatula.n_samples)
    points = blade_points(tip_linear, tool_axis_linear, spatula.length, samples)
    depth = (points - b_vial) @ bore                                  # (T, K)
    submerged = depth < 0
    allowed = wall_radius(depth, settings.vial) - spatula.radius - settings.margin
    active, fraction, _ = lip_crossing(tip_linear, tool_axis_linear, vial_frame,
                                       spatula.length)
    lip_allowed = settings.vial.lip_radius - spatula.radius - settings.margin

    xi = cp.Variable((horizon, n_state))
    u = cp.Variable((horizon - 1, n_ctrl))

    constraints = [xi[0] == x0]
    for t in range(horizon - 1):
        constraints.append(xi[t + 1] == A_dyn @ xi[t] + B_dyn @ u[t])
    end_steps = max(0, min(int(anchor_end_steps), horizon))
    for t in range(horizon - end_steps, horizon):
        constraints.append(xi[t, 0:3] == anchor_target)

    def blade_point(t, s):
        """Blade point at ``s`` as an affine function of the state at ``t``."""
        offset = s * spatula.length
        point = xi[t, 0:3] - offset * tool_axis_linear[t]
        return point - offset * (tool_jacobian[t] @ (xi[t, 3:6] - eta_linear[t]))

    for t in range(horizon):
        for k, s in enumerate(samples):
            if submerged[t, k]:
                constraints.append(
                    cp.norm(cross_section @ (blade_point(t, s) - b_vial), 2)
                    <= float(allowed[t, k]))
        if active[t]:
            constraints.append(
                cp.norm(cross_section @ (blade_point(t, float(fraction[t])) - b_vial), 2)
                <= lip_allowed)

    cost = cp.sum([cp.quad_form(xi[t] - mu_seq[t], cp.psd_wrap(weight[t]))
                   for t in range(horizon)])
    cost += control_cost * cp.sum_squares(u)
    problem = cp.Problem(cp.Minimize(cost), constraints)

    def attempt(solver, **options) -> bool:
        try:
            problem.solve(solver=solver, verbose=settings.verbose, **options)
        except cp.error.SolverError:
            return False
        return problem.status in ("optimal", "optimal_inaccurate")

    if settings.solver == "CLARABEL":
        # Tight enough that the wall clearance is settled to well under a micrometre,
        # loose enough that the solver can certify it and report a plain optimum.
        solved = attempt("CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-7,
                         tol_feas=1e-7, max_iter=2000)
    else:
        solved = attempt(settings.solver)
    if not solved and "SCS" in cp.installed_solvers():
        solved = attempt("SCS", max_iters=30000, eps=1e-8)
    if not solved and "ECOS" in cp.installed_solvers():
        solved = attempt("ECOS", max_iters=4000, abstol=1e-8)

    return (xi.value if solved else None), str(problem.status), solved


def worst_violation(xi, vial_frame, q_ref, settings: SafetySettings) -> float:
    """How far outside the vial a trajectory reaches, in metres; zero or less if inside."""
    quat = ori.from_chart(xi[:, 3:6], q_ref)
    wall, lip = containment_margins(xi[:, :3], quat, vial_frame, settings)
    clearances = np.concatenate([wall, lip])
    clearances = clearances[np.isfinite(clearances)]
    return float(-clearances.min()) if clearances.size else 0.0


def constrain_to_vial(xi, *, A_dyn, B_dyn, x0, mu_seq, sigma_seq, control_cost,
                      vial_frame, q_ref, settings: SafetySettings,
                      anchor_end_steps: int = 0, anchor_target=None
                      ) -> Tuple[np.ndarray, str]:
    """Re-solve a trajectory under the vial's geometry.

    ``xi`` is the unconstrained trajectory, used as the point the spatula axis is
    linearised about. The solve is repeated, each time relinearising about its own
    result, until the orientation settles or ``settings.max_iterations`` is
    reached. The stroke that comes out is then checked against the true geometry,
    since the constraints themselves were imposed on a linearisation.

    Returns the constrained trajectory and the solver's status. Raises
    :class:`InfeasibleTrajectory` when no solution satisfies the constraints or when
    the solution does not survive that check.
    """
    if anchor_target is None:
        anchor_target = np.asarray(vial_frame[1], float)
    xi = np.asarray(xi, float)

    start_violation = worst_violation(xi[:1], vial_frame, q_ref, settings)
    if start_violation > settings.containment_tolerance:
        raise InfeasibleTrajectory(
            f"the stroke is required to start {start_violation * 1e3:.1f} mm outside the "
            "vial, so no trajectory from there can stay inside it; move the start pose in",
            0)

    previous_eta = xi[:, 3:6].copy()
    status = ""

    for iteration in range(max(1, int(settings.max_iterations))):
        eta_linear = xi[:, 3:6]
        quat_linear = ori.from_chart(eta_linear, q_ref)
        solution, status, solved = _solve_socp(
            A_dyn=A_dyn, B_dyn=B_dyn, x0=x0, mu_seq=mu_seq, sigma_seq=sigma_seq,
            control_cost=control_cost, vial_frame=vial_frame,
            tip_linear=xi[:, :3],
            tool_axis_linear=np.atleast_2d(tool_axis_world(quat_linear)),
            eta_linear=eta_linear,
            tool_jacobian=ori.tool_axis_jacobian(eta_linear, q_ref),
            settings=settings, anchor_end_steps=anchor_end_steps,
            anchor_target=np.asarray(anchor_target, float))
        if not solved:
            raise InfeasibleTrajectory(status, iteration)

        xi = np.asarray(solution)
        change = float(np.linalg.norm(xi[:, 3:6] - previous_eta, axis=1).max())
        previous_eta = xi[:, 3:6].copy()
        if change < float(settings.tolerance_rad):
            break

    violation = worst_violation(xi, vial_frame, q_ref, settings)
    if violation > UNUSABLE_SOLUTION_M:
        raise InfeasibleTrajectory(
            f"the solver reported {status} but returned a trajectory {violation * 1e3:.0f} mm "
            "outside the vial, which is what an infeasible problem looks like here; the "
            "stroke asked for cannot be made to fit", settings.max_iterations - 1)
    if violation > settings.containment_tolerance:
        raise InfeasibleTrajectory(
            f"{status}, but the stroke reaches {violation * 1e3:.3f} mm outside the vial "
            "once its own orientation is taken into account", settings.max_iterations - 1)

    return xi, status
