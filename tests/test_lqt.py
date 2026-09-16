"""Tests for the batch linear quadratic tracker.

The system and its lifted form are checked against hand-written matrices and
against a step-by-step roll-out; the tracker is checked against the analytic
minimiser of the lifted cost and against the trade-off the control weight sets.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

from tplqt.lqt import canonical_system, lifted_matrices, solve_lqt

DT = 0.02


def sine_reference(horizon: int = 60, dt: float = DT):
    """A dynamically consistent position and velocity reference in two coordinates."""
    t = np.arange(horizon) * dt
    w = 2 * np.pi / (horizon * dt)
    pos = np.column_stack([0.10 * np.sin(w * t), 0.05 * np.cos(w * t)])
    vel = np.column_stack([0.10 * w * np.cos(w * t), -0.05 * w * np.sin(w * t)])
    return np.hstack([pos, vel])


def control_effort(xi, dt: float = DT) -> float:
    """Squared control effort read back off a double-integrator state sequence."""
    u = np.diff(xi[:, 2:], axis=0) / dt
    return float((u ** 2).sum())


@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_double_integrator_matrices_are_exact(n_dim, dt):
    """For two derivatives the system is [[I, dt I], [0, I]] with B = [dt^2/2 I, dt I]'."""
    eye = np.eye(n_dim)
    zero = np.zeros((n_dim, n_dim))
    A, B = canonical_system(n_dim, n_deriv=2, dt=dt)
    np.testing.assert_allclose(A, np.block([[eye, dt * eye], [zero, eye]]), rtol=0, atol=0)
    np.testing.assert_allclose(B, np.vstack([dt ** 2 / 2 * eye, dt * eye]), rtol=0, atol=0)


def test_triple_integrator_follows_the_taylor_series():
    """For three derivatives each block is dt^k / k! above the diagonal."""
    dt, n_dim = 0.1, 2
    eye = np.eye(n_dim)
    zero = np.zeros((n_dim, n_dim))
    A, B = canonical_system(n_dim, n_deriv=3, dt=dt)
    np.testing.assert_allclose(A, np.block([[eye, dt * eye, dt ** 2 / 2 * eye],
                                            [zero, eye, dt * eye],
                                            [zero, zero, eye]]), rtol=0, atol=1e-18)
    np.testing.assert_allclose(B, np.vstack([dt ** 3 / 6 * eye, dt ** 2 / 2 * eye, dt * eye]),
                               rtol=0, atol=1e-18)


@pytest.mark.parametrize("n_dim, n_deriv", [(1, 2), (3, 2), (3, 3), (6, 4)])
def test_system_shapes_follow_the_state_layout(n_dim, n_deriv):
    """The state stacks n_deriv blocks of n_dim coordinates and the control drives one."""
    A, B = canonical_system(n_dim, n_deriv=n_deriv, dt=DT)
    assert A.shape == (n_dim * n_deriv, n_dim * n_deriv)
    assert B.shape == (n_dim * n_deriv, n_dim)


def test_constant_input_integrates_to_the_analytic_double_integrator():
    """Stepping a constant u from rest gives x_t = dt^2 t^2 u / 2 and v_t = dt t u."""
    dt, n_dim, horizon = 0.05, 2, 40
    A, B = canonical_system(n_dim, n_deriv=2, dt=dt)
    u = np.array([0.3, -0.7])
    x = np.zeros(2 * n_dim)
    states = [x]
    for _ in range(horizon):
        x = A @ x + B @ u
        states.append(x)
    states = np.array(states)
    t = np.arange(horizon + 1)
    np.testing.assert_allclose(states[:, :n_dim], 0.5 * (dt * t)[:, None] ** 2 * u,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(states[:, n_dim:], (dt * t)[:, None] * u,
                               rtol=1e-12, atol=1e-15)


def test_lifted_form_starts_at_the_initial_state():
    """The first block row of S_x is the identity and the first of S_u is zero."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    n_state = A.shape[0]
    S_x, S_u = lifted_matrices(A, B, horizon=7)
    np.testing.assert_allclose(S_x[:n_state], np.eye(n_state), rtol=0, atol=0)
    np.testing.assert_allclose(S_u[:n_state], 0.0, rtol=0, atol=0)


def test_lifted_control_blocks_are_shifted_system_powers():
    """Block (t, j) of S_u is A^(t-j-1) B below the diagonal and zero on or above it."""
    A, B = canonical_system(2, n_deriv=3, dt=DT)
    horizon = 6
    n_state, n_ctrl = B.shape
    _, S_u = lifted_matrices(A, B, horizon)
    for t in range(horizon):
        for j in range(horizon):
            block = S_u[t * n_state:(t + 1) * n_state, j * n_ctrl:(j + 1) * n_ctrl]
            expected = (np.linalg.matrix_power(A, t - j - 1) @ B if j < t
                        else np.zeros_like(block))
            np.testing.assert_allclose(block, expected, rtol=1e-12, atol=1e-15)


def test_lifted_rollout_matches_stepwise_integration():
    """S_x x0 + S_u u equals the trajectory obtained by stepping the same controls."""
    rng = np.random.default_rng(3)
    A, B = canonical_system(3, n_deriv=2, dt=DT)
    horizon = 25
    n_state, n_ctrl = B.shape
    x0 = rng.normal(size=n_state)
    u = rng.normal(size=(horizon, n_ctrl))
    S_x, S_u = lifted_matrices(A, B, horizon)
    lifted = (S_x @ x0 + S_u @ u.ravel()).reshape(horizon, n_state)

    x = x0.copy()
    stepped = []
    for t in range(horizon):
        stepped.append(x)
        x = A @ x + B @ u[t]
    np.testing.assert_allclose(lifted, np.array(stepped), rtol=1e-12, atol=1e-14)


def test_solution_starts_exactly_at_the_initial_state():
    """The tracked trajectory begins at x0 whatever the reference asks for."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    mu = sine_reference()
    sigma = np.tile(1e-6 * np.eye(4), (len(mu), 1, 1))
    x0 = mu[0] + np.array([0.02, -0.03, 0.1, 0.1])
    xi = solve_lqt(A, B, x0, mu, sigma, control_cost=1e-6)
    np.testing.assert_array_equal(xi[0], x0)


def test_tight_reference_is_tracked_to_a_fraction_of_a_millimetre():
    """A tight covariance and a small control cost hold the state on the reference."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    mu = sine_reference()
    sigma = np.tile(1e-6 * np.eye(4), (len(mu), 1, 1))
    xi = solve_lqt(A, B, mu[0], mu, sigma, control_cost=1e-6)
    error = np.linalg.norm(xi[:, :2] - mu[:, :2], axis=1)
    assert error.max() < 2e-4


def test_control_cost_trades_tracking_against_effort():
    """Raising the control weight monotonically worsens tracking and lowers effort."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    mu = sine_reference()
    sigma = np.tile(1e-4 * np.eye(4), (len(mu), 1, 1))
    costs = [1e-4, 1.0, 1e2, 1e4]
    errors, efforts = [], []
    for r in costs:
        xi = solve_lqt(A, B, mu[0], mu, sigma, control_cost=r)
        errors.append(float(np.sqrt(((xi[:, :2] - mu[:, :2]) ** 2).sum(axis=1).mean())))
        efforts.append(control_effort(xi))
    assert all(a < b for a, b in zip(errors, errors[1:]))
    assert all(a > b for a, b in zip(efforts, efforts[1:]))
    assert errors[0] < 1e-4 and errors[-1] > 0.1
    assert efforts[-1] < 0.05 * efforts[0]


def test_constant_reference_is_reached_and_held():
    """A tightly weighted fixed point is reached and the state stays there at rest."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    horizon = 60
    target = np.array([0.05, -0.02, 0.0, 0.0])
    mu = np.tile(target, (horizon, 1))
    sigma = np.tile(np.diag([1e-8, 1e-8, 1e-6, 1e-6]), (horizon, 1, 1))
    xi = solve_lqt(A, B, np.zeros(4), mu, sigma, control_cost=1e-6)
    tail = xi[-10:]
    np.testing.assert_allclose(tail[:, :2], np.tile(target[:2], (len(tail), 1)),
                               rtol=0, atol=1e-4)
    assert np.abs(tail[:, 2:]).max() < 1e-3
    assert np.ptp(tail[:, :2], axis=0).max() < 1e-4


def test_matches_the_analytic_lifted_minimiser():
    """On a two-step horizon the solution equals the explicitly built normal-equation one."""
    dt, r = 0.05, 1e-3
    A, B = canonical_system(1, n_deriv=2, dt=dt)
    x0 = np.array([0.05, -0.10])
    mu = np.array([[0.10, 0.0], [0.20, 0.30]])
    sigma = np.stack([np.diag([1e-3, 1e-2]), np.diag([2e-4, 5e-3])])

    S_x = np.vstack([np.eye(2), A])
    S_u = np.zeros((4, 2))
    S_u[2:4, 0:1] = B
    Q = block_diag(np.linalg.inv(sigma[0]), np.linalg.inv(sigma[1]))
    u = np.linalg.solve(S_u.T @ Q @ S_u + r * np.eye(2),
                        S_u.T @ Q @ (mu.ravel() - S_x @ x0))
    expected = (S_x @ x0 + S_u @ u).reshape(2, 2)

    xi = solve_lqt(A, B, x0, mu, sigma, control_cost=r)
    np.testing.assert_allclose(xi, expected, rtol=1e-10, atol=1e-12)


def test_solution_is_equivariant_under_translation():
    """Translating the reference and the initial state translates the solution."""
    A, B = canonical_system(2, n_deriv=2, dt=DT)
    mu = sine_reference()
    sigma = np.tile(1e-4 * np.eye(4), (len(mu), 1, 1))
    delta = np.array([0.37, -0.21])
    x0 = mu[0] + np.array([0.01, 0.01, 0.0, 0.0])

    xi = solve_lqt(A, B, x0, mu, sigma, control_cost=1e-3)
    mu_shifted = mu + np.concatenate([delta, np.zeros(2)])
    x0_shifted = x0 + np.concatenate([delta, np.zeros(2)])
    xi_shifted = solve_lqt(A, B, x0_shifted, mu_shifted, sigma, control_cost=1e-3)

    shift = np.concatenate([delta, np.zeros(2)])
    np.testing.assert_allclose(xi_shifted - xi, np.tile(shift, (len(xi), 1)),
                               rtol=0, atol=1e-10)
