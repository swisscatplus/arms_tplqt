"""Gaussian mixtures and the operations the task-parameterised model is built from.

A :class:`GaussianMixture` is a set of ``K`` Gaussians over a common ``D``-dimensional
space together with mixing weights. Three operations are needed by the
task-parameterised pipeline:

marginal
    keep a contiguous block of dimensions, which extracts one task frame's view out
    of the stacked per-frame observation;
transform
    push a mixture through an affine map ``x -> A x + b``, which expresses a frame's
    Gaussians in world coordinates;
product
    multiply two mixtures component by component, which fuses the frames into a
    single Gaussian per state (the product of Gaussians of task-parameterised
    models).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class GaussianMixture:
    """``K`` Gaussians over ``R^D`` with mixing weights.

    Parameters
    ----------
    mu : (K, D) array
        Component means.
    sigma : (K, D, D) array
        Component covariances.
    priors : (K,) array, optional
        Mixing weights; uniform when omitted.
    """

    def __init__(self, mu, sigma, priors: Optional[np.ndarray] = None):
        self.mu = np.asarray(mu, float)
        self.sigma = np.asarray(sigma, float)
        if self.mu.ndim != 2 or self.sigma.ndim != 3:
            raise ValueError(f"expected mu (K, D) and sigma (K, D, D), "
                             f"got {self.mu.shape} and {self.sigma.shape}")
        if self.sigma.shape[:2] != self.mu.shape or self.sigma.shape[2] != self.mu.shape[1]:
            raise ValueError(f"mu {self.mu.shape} and sigma {self.sigma.shape} disagree")
        K = self.mu.shape[0]
        self.priors = (np.full(K, 1.0 / K) if priors is None
                       else np.asarray(priors, float))
        self._precision: Optional[np.ndarray] = None

    @property
    def n_states(self) -> int:
        return int(self.mu.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.mu.shape[1])

    @property
    def precision(self) -> np.ndarray:
        """(K, D, D) inverse covariances, computed on first use."""
        if self._precision is None:
            self._precision = np.linalg.inv(self.sigma)
        return self._precision

    def marginal(self, dims: slice) -> "GaussianMixture":
        """The mixture over a contiguous block of dimensions."""
        return GaussianMixture(self.mu[:, dims], self.sigma[:, dims, dims], self.priors)

    def transform(self, A, b) -> "GaussianMixture":
        """The mixture of ``A x + b`` for ``x`` distributed as this mixture."""
        A = np.asarray(A, float)
        b = np.asarray(b, float)
        mu = self.mu @ A.T + b
        sigma = np.einsum("ij,ajk,lk->ail", A, self.sigma, A)
        return GaussianMixture(mu, sigma, self.priors)

    def __mul__(self, other: "GaussianMixture") -> "GaussianMixture":
        """Component-wise product of Gaussians.

        For each state ``k`` the product of two Gaussians is Gaussian with
        precision ``P_k = P1_k + P2_k`` and mean ``mu_k = P_k^-1 (P1_k mu1_k +
        P2_k mu2_k)``. Mixing weights are taken from the left operand.
        """
        if self.n_states != other.n_states or self.n_dim != other.n_dim:
            raise ValueError(f"cannot multiply mixtures of shapes {self.mu.shape} "
                             f"and {other.mu.shape}")
        precision = self.precision + other.precision
        info = (np.einsum("aij,aj->ai", self.precision, self.mu)
                + np.einsum("aij,aj->ai", other.precision, other.mu))
        sigma = np.linalg.inv(precision)
        mu = np.einsum("aij,aj->ai", sigma, info)
        prod = GaussianMixture(mu, sigma, self.priors)
        prod._precision = precision
        return prod

    def log_likelihoods(self, x) -> np.ndarray:
        """(K, T) log densities of the rows of ``x`` under each component."""
        x = np.atleast_2d(np.asarray(x, float))
        dx = self.mu[:, None, :] - x[None, :, :]                       # (K, T, D)
        maha = np.einsum("ktd,kde,kte->kt", dx, self.precision, dx)
        _, logdet = np.linalg.slogdet(self.sigma)                      # (K,)
        return -0.5 * (maha + self.n_dim * np.log(2 * np.pi) + logdet[:, None])

    def sequence(self, state_seq):
        """Per-timestep means and covariances for a sequence of state indices.

        Returns ``(mu_seq, sigma_seq)`` of shapes ``(T, D)`` and ``(T, D, D)`` --
        the block-diagonal trajectory distribution the linear quadratic tracker
        tracks, kept in block form rather than assembled densely.
        """
        idx = np.asarray(state_seq, int)
        return self.mu[idx], self.sigma[idx]
