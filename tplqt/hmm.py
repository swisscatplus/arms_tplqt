"""Hidden Markov model with Gaussian emissions, fitted by Baum-Welch.

The model is fitted on several demonstrations at once: each is a sequence of
observations and all of them share one set of ``K`` Gaussian emission densities,
one transition matrix and one initial-state distribution.

States are initialised by splitting every demonstration into ``K`` equal-length
time bins, which gives the states a common temporal ordering across
demonstrations before the expectation-maximisation refines them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .gaussian import GaussianMixture

_TINY = np.finfo(float).tiny
_LARGEST = np.finfo(float).max


class HiddenMarkovModel:
    """Gaussian-emission HMM over a set of demonstrations.

    Attributes
    ----------
    emissions : GaussianMixture
        The ``K`` emission densities; ``emissions.priors`` holds the state
        occupancies estimated by the last expectation step.
    trans : (K, K) array
        Row-stochastic transition matrix, ``trans[i, j] = p(s_t = j | s_t-1 = i)``.
    init_priors : (K,) array
        Distribution of the first state.
    """

    def __init__(self, emissions: GaussianMixture, trans, init_priors):
        self.emissions = emissions
        self.trans = np.asarray(trans, float)
        self.init_priors = np.asarray(init_priors, float)

    @property
    def n_states(self) -> int:
        return self.emissions.n_states

    @property
    def n_dim(self) -> int:
        return self.emissions.n_dim

    def copy(self) -> "HiddenMarkovModel":
        """An independent copy of the model."""
        emissions = GaussianMixture(self.emissions.mu.copy(), self.emissions.sigma.copy(),
                                    self.emissions.priors.copy())
        return HiddenMarkovModel(emissions, self.trans.copy(), self.init_priors.copy())

    @classmethod
    def from_time_bins(cls, sequences: Sequence[np.ndarray], n_states: int, *,
                       reg: float = 1e-8) -> "HiddenMarkovModel":
        """Initialise from equal-length time bins of each sequence.

        Bin ``i`` of every sequence feeds state ``i``, so the states start out
        ordered along the demonstrated motion. Transitions are initialised as a
        left-to-right skeleton over a small uniform floor and the initial-state
        distribution uniform. The skeleton is a starting point for the forward
        recursion, not a distribution -- its rows sum to slightly more than one, and
        the first maximisation step replaces it with a proper stochastic matrix.
        """
        seqs = [np.asarray(s, float) for s in sequences]
        if not seqs:
            raise ValueError("no sequences to fit")
        n_dim = seqs[0].shape[1]
        edges = [np.round(np.linspace(0, len(s), n_states + 1)).astype(int) for s in seqs]

        mu = np.zeros((n_states, n_dim))
        sigma = np.zeros((n_states, n_dim, n_dim))
        counts = np.zeros(n_states)
        for i in range(n_states):
            binned = np.concatenate([s[e[i]:e[i + 1]] for s, e in zip(seqs, edges)])
            if len(binned) < 2:
                raise ValueError(
                    f"time bin {i} of {n_states} holds {len(binned)} samples, too few to "
                    "give a mean and a covariance. Use fewer states or longer sequences")
            counts[i] = len(binned)
            mu[i] = binned.mean(axis=0)
            sigma[i] = np.cov(binned.T) + reg * np.eye(n_dim)

        mean_length = float(np.mean([len(s) for s in seqs]))
        forward = n_states / mean_length
        trans = np.full((n_states, n_states), 0.01)
        for i in range(n_states - 1):
            trans[i, i] = 1.0 - forward
            trans[i, i + 1] = forward
        trans[-1, -1] = 1.0

        emissions = GaussianMixture(mu, sigma, counts / counts.sum())
        init_priors = np.full(n_states, 1.0 / n_states)
        return cls(emissions, trans, init_priors)

    @classmethod
    def fit(cls, sequences: Sequence[np.ndarray], n_states: int, *, reg: float = 1e-6,
            max_iter: int = 100, min_iter: int = 2, tol: float = 1e-4,
            init_reg: float = 1e-8, left_to_right: bool = False,
            verbose: bool = False) -> "HiddenMarkovModel":
        """Fit by Baum-Welch on several sequences.

        Parameters
        ----------
        sequences : list of (T_n, D) arrays
            The demonstrations; they may differ in length.
        n_states : int
            Number of hidden states.
        reg : float
            Variance added to the diagonal of every covariance at each
            maximisation step, which keeps the covariances well conditioned.
        max_iter, min_iter, tol : int, int, float
            Iteration budget and the convergence test on the average log-likelihood
            per demonstration: iteration stops once it improves by less than ``tol``,
            and, if the variance floor makes it fall, on the better of the two
            iterates.
        init_reg : float
            Variance added to the time-bin covariances at initialisation.
        left_to_right : bool
            Restrict transitions to self and next state.
        """
        seqs = [np.asarray(s, float) for s in sequences]
        model = cls.from_time_bins(seqs, n_states, reg=init_reg)
        data = np.concatenate(seqs)                                    # (N, D)
        eye = np.eye(model.n_dim)

        mask = None
        if left_to_right:
            mask = np.eye(n_states) + np.diag(np.ones(n_states - 1), 1)

        previous_ll = -np.inf
        best = None
        for iteration in range(max_iter):
            gammas, zetas, log_liks = [], [], []
            for seq in seqs:
                gamma, zeta, log_lik = model._forward_backward(seq)
                gammas.append(gamma)
                zetas.append(zeta)
                log_liks.append(log_lik)

            gamma = np.hstack(gammas)                                  # (K, N)
            zeta = np.dstack(zetas)                                    # (K, K, N - n_seq)
            gamma_first = np.hstack([g[:, :1] for g in gammas])        # (K, n_seq)
            gamma_but_last = np.hstack([g[:, :-1] for g in gammas])    # (K, N - n_seq)
            weights = gamma / (gamma.sum(axis=1, keepdims=True) + _TINY)

            mu = weights @ data                                        # (K, D)
            sigma = np.empty((n_states, model.n_dim, model.n_dim))
            for k in range(n_states):
                centred = data - mu[k]                                 # (N, D)
                sigma[k] = (centred * weights[k, :, None]).T @ centred + reg * eye

            model.emissions = GaussianMixture(mu, sigma, gamma.mean(axis=1))
            model.init_priors = gamma_first.mean(axis=1)
            model.trans = (zeta.sum(axis=2)
                           / (gamma_but_last.sum(axis=1)[:, None] + _TINY))
            if mask is not None:
                model.trans *= mask
                model.trans /= model.trans.sum(axis=1, keepdims=True)

            log_lik = float(np.mean(log_liks))
            if verbose:
                print(f"iteration {iteration}: log-likelihood {log_lik:.6f}")
            if iteration > min_iter and log_lik < previous_ll:
                # Once the model has settled, the variance floor can outweigh the fit.
                # Stop there and keep the better of the two iterates, not the later one.
                return best
            if iteration > min_iter and log_lik - previous_ll < tol:
                return model
            previous_ll = log_lik
            best = model.copy()

        return model

    def _forward_backward(self, obs):
        """Scaled forward-backward pass over one sequence.

        Returns the state posteriors ``gamma`` (K, T), the transition posteriors
        ``zeta`` (K, K, T-1) and the sequence log-likelihood.
        """
        log_b = self.emissions.log_likelihoods(obs)                    # (K, T)
        offset = log_b.max(axis=0)                                     # (T,)
        b = np.exp(log_b - offset)
        K, T = b.shape

        alpha = np.zeros((K, T))
        scale = np.zeros(T)
        alpha[:, 0] = self.init_priors * b[:, 0]
        scale[0] = 1.0 / (alpha[:, 0].sum() + _TINY)
        alpha[:, 0] *= scale[0]
        for t in range(1, T):
            alpha[:, t] = (alpha[:, t - 1] @ self.trans) * b[:, t]
            scale[t] = 1.0 / (alpha[:, t].sum() + _TINY)
            alpha[:, t] *= scale[t]

        beta = np.zeros((K, T))
        beta[:, -1] = scale[-1]
        for t in range(T - 2, -1, -1):
            # Capped so that a sequence the model fits badly degrades into a flat
            # posterior rather than into infinities and then not-a-numbers.
            beta[:, t] = np.minimum(scale[t] * (self.trans @ (beta[:, t + 1] * b[:, t + 1])),
                                    _LARGEST)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0) + _TINY
        zeta = (self.trans[:, :, None] * alpha[:, None, :-1]
                * (b[None, :, 1:] * beta[None, :, 1:]))
        log_lik = float(-np.sum(np.log(scale)) + offset.sum())
        return gamma, zeta, log_lik

    def state_posteriors(self, obs) -> np.ndarray:
        """(K, T) posterior probability of each state along ``obs``."""
        gamma, _, _ = self._forward_backward(obs)
        return gamma

    def viterbi(self, obs) -> np.ndarray:
        """(T,) most likely state sequence for ``obs``."""
        log_b = self.emissions.log_likelihoods(obs)                    # (K, T)
        log_trans = np.log(self.trans + _TINY)
        K, T = log_b.shape

        delta = np.zeros((K, T))
        psi = np.zeros((K, T), int)
        delta[:, 0] = np.log(self.init_priors + _TINY) + log_b[:, 0]
        for t in range(1, T):
            scores = delta[:, t - 1, None] + log_trans                 # (K_from, K_to)
            psi[:, t] = np.argmax(scores, axis=0)
            delta[:, t] = scores[psi[:, t], np.arange(K)] + log_b[:, t]

        seq = np.zeros(T, int)
        seq[-1] = int(np.argmax(delta[:, -1]))
        for t in range(T - 2, -1, -1):
            seq[t] = psi[seq[t + 1], t + 1]
        return seq
