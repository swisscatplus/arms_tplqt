"""Batch linear quadratic tracking for a chain of integrators.

The state is a configuration and its derivatives, ``xi = [x, xdot, ...]``, driven by
the discrete-time system ``xi_{t+1} = A xi_t + B u_t``. Stacking the whole
trajectory gives the lifted form

    xi = S_x xi_0 + S_u u,

so tracking a per-timestep Gaussian reference ``(mu_t, Sigma_t)`` is the
unconstrained quadratic problem

    min_u  sum_t (xi_t - mu_t)' Sigma_t^-1 (xi_t - mu_t) + r * u' u,

whose solution follows from one linear system. Solving in the lifted form (rather
than by a Riccati recursion) is what lets the same reference be handed to the
constrained solver in :mod:`tplqt.safety`, which adds geometric constraints to
exactly this cost.
"""
from __future__ import annotations

from math import factorial

import numpy as np
from scipy.linalg import solve as solve_linear


def canonical_system(n_dim: int, n_deriv: int = 2, dt: float = 0.01):
    """Discrete-time chain of ``n_deriv`` integrators over ``n_dim`` coordinates.

    Returns ``(A, B)`` of shapes ``(n_dim * n_deriv, n_dim * n_deriv)`` and
    ``(n_dim * n_deriv, n_dim)``. The state stacks the derivatives in blocks,
    ``[x, xdot, ...]``, and the control drives the highest derivative. With
    ``n_deriv = 2`` this is the double integrator with ``A = [[I, dt I], [0, I]]``
    and ``B = [dt^2 / 2 I, dt I]'``.
    """
    a = np.zeros((n_deriv, n_deriv))
    for i in range(n_deriv):
        a += np.diag(np.ones(n_deriv - i), i) * dt ** i / factorial(i)
    b = np.zeros((n_deriv, 1))
    for i in range(1, n_deriv + 1):
        b[n_deriv - i] = dt ** i / factorial(i)
    return np.kron(a, np.eye(n_dim)), np.kron(b, np.eye(n_dim))


def lifted_matrices(A, B, horizon: int):
    """Lifted transfer matrices ``(S_x, S_u)`` with ``xi = S_x xi_0 + S_u u``.

    ``S_x`` stacks ``A^t`` for ``t = 0 .. T-1`` and ``S_u`` is block lower
    triangular with ``S_u[t, j] = A^(t-j-1) B`` for ``j < t``. The first state is
    therefore ``xi_0`` itself, and the last control has no effect on the
    trajectory.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n_state, n_ctrl = B.shape
    T = int(horizon)

    S_x = np.zeros((n_state * T, n_state))
    powers = []
    A_t = np.eye(n_state)
    for t in range(T):
        S_x[t * n_state:(t + 1) * n_state] = A_t
        powers.append(A_t @ B)
        A_t = A_t @ A

    S_u = np.zeros((n_state * T, n_ctrl * T))
    for t in range(T):
        for j in range(t):
            S_u[t * n_state:(t + 1) * n_state,
                j * n_ctrl:(j + 1) * n_ctrl] = powers[t - j - 1]
    return S_x, S_u


def solve_lqt(A, B, x0, mu_seq, sigma_seq, control_cost: float):
    """Track a Gaussian reference from ``x0`` and return the state sequence.

    Parameters
    ----------
    A, B : arrays
        Discrete-time system, as returned by :func:`canonical_system`.
    x0 : (n_state,) array
        Initial state; the returned trajectory starts exactly there.
    mu_seq : (T, n_state) array
        Reference mean per timestep.
    sigma_seq : (T, n_state, n_state) array
        Reference covariance per timestep; its inverse weights the tracking error,
        so a tight covariance is a tightly tracked reference.
    control_cost : float
        Weight ``r`` on the squared control effort, strictly positive. Raising it
        trades tracking accuracy for a smoother trajectory. It is what makes the
        problem strictly convex: the last control has no effect on the trajectory,
        so without it the system to solve is singular.

    Returns
    -------
    (T, n_state) array
        The tracked state sequence.
    """
    if control_cost <= 0.0:
        raise ValueError("control_cost must be positive; the last control does not move "
                         "the trajectory, so a zero cost leaves it undetermined")
    x0 = np.asarray(x0, float)
    mu_seq = np.asarray(mu_seq, float)
    sigma_seq = np.asarray(sigma_seq, float)
    T, n_state = mu_seq.shape
    n_ctrl = np.asarray(B).shape[1]

    S_x, S_u = lifted_matrices(A, B, T)
    Q = np.linalg.inv(sigma_seq)                                   # (T, n_state, n_state)

    error = mu_seq - (S_x @ x0).reshape(T, n_state)                # (T, n_state)
    S_u_blocks = S_u.reshape(T, n_state, n_ctrl * T)
    Q_S_u = np.einsum("tij,tjc->tic", Q, S_u_blocks).reshape(T * n_state, n_ctrl * T)
    Q_error = np.einsum("tij,tj->ti", Q, error).ravel()

    hessian = S_u.T @ Q_S_u + control_cost * np.eye(n_ctrl * T)
    hessian = 0.5 * (hessian + hessian.T)          # symmetric up to rounding by construction
    gradient = S_u.T @ Q_error
    u = solve_linear(hessian, gradient, assume_a="pos")
    return (S_u @ u + S_x @ x0).reshape(T, n_state)
