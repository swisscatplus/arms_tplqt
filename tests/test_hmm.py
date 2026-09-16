"""Tests for the Gaussian-emission hidden Markov model.

The sequences used here pass through a few well-separated clusters in a fixed
order, so the parameters the fit should recover are known in closed form: the
cluster centres are the emission means, and the transition probabilities are the
counts of the state changes the sequences make.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tplqt.gaussian import GaussianMixture
from tplqt.hmm import HiddenMarkovModel

CENTRES = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
DURATIONS = (20, 60, 40)
N_SEQUENCES = 4


def cluster_sequences(durations=DURATIONS, noise=0.05, n_sequences=N_SEQUENCES):
    """Sequences visiting ``CENTRES`` in order, each centre held for a fixed time."""
    return [np.concatenate([c + noise * np.random.default_rng(100 + s + 7 * i)
                            .normal(size=(d, 2))
                            for i, (c, d) in enumerate(zip(CENTRES, durations))])
            for s in range(n_sequences)]


@pytest.fixture(scope="module")
def clusters():
    return cluster_sequences()


@pytest.fixture(scope="module")
def fitted(clusters):
    return HiddenMarkovModel.fit(clusters, 3)


# Sequences of length 10 and 7 split into 4 bins by ``round(linspace(0, T, 5))``.
RAMPS = [np.stack([np.arange(10.0), 0.1 * np.arange(10.0) ** 2], axis=1),
         np.stack([np.arange(7.0) + 0.5, -0.2 * np.arange(7.0)], axis=1)]
EDGES = [[0, 2, 5, 8, 10], [0, 2, 4, 5, 7]]


def test_time_bin_means_are_the_pooled_bin_means():
    """Each initial state is the mean of its time bin pooled over the sequences."""
    model = HiddenMarkovModel.from_time_bins(RAMPS, 4)
    expected = [np.concatenate([s[e[i]:e[i + 1]] for s, e in zip(RAMPS, EDGES)]).mean(axis=0)
                for i in range(4)]
    assert_allclose(model.emissions.mu, expected, rtol=0, atol=1e-12)


def test_time_bin_weights_are_the_sample_fractions():
    """The initial mixture weights are the fraction of samples falling in each bin."""
    model = HiddenMarkovModel.from_time_bins(RAMPS, 4)
    counts = np.array([sum(e[i + 1] - e[i] for e in EDGES) for i in range(4)], float)
    assert_allclose(model.emissions.priors, counts / 17.0, rtol=0, atol=1e-12)
    assert model.emissions.priors.sum() == pytest.approx(1.0, abs=1e-12)


def test_time_bin_transitions_are_left_to_right():
    """Initial transitions put all but a small floor on the self and next state."""
    model = HiddenMarkovModel.from_time_bins(RAMPS, 4)
    forward = 4.0 / np.mean([len(s) for s in RAMPS])
    for i in range(3):
        assert model.trans[i, i] == pytest.approx(1.0 - forward, abs=1e-12)
        assert model.trans[i, i + 1] == pytest.approx(forward, abs=1e-12)
        assert model.trans[i, i] + model.trans[i, i + 1] == pytest.approx(1.0, abs=1e-12)
    assert model.trans[3, 3] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diag(model.trans) > 0.0) and np.all(np.diag(model.trans, 1) > 0.0)
    assert model.trans[2, 0] == pytest.approx(0.01, abs=1e-12)


def test_fit_recovers_the_cluster_centres(fitted):
    """Every fitted mean matches a distinct generating centre to within the noise."""
    distance = np.linalg.norm(fitted.emissions.mu[:, None, :] - CENTRES[None, :, :], axis=2)
    nearest = distance.argmin(axis=1)
    assert sorted(nearest.tolist()) == [0, 1, 2]
    assert_allclose(fitted.emissions.mu, CENTRES[nearest], rtol=0, atol=0.01)


def test_fit_recovers_the_generating_transition_probabilities(fitted):
    """Self-transition probabilities match the counted (d - 1) / d of each cluster."""
    expected = np.zeros((3, 3))
    for i, d in enumerate(DURATIONS[:-1]):
        expected[i, i], expected[i, i + 1] = (d - 1.0) / d, 1.0 / d
    expected[2, 2] = 1.0
    assert_allclose(fitted.trans, expected, rtol=0, atol=1e-8)


def test_fitted_distributions_are_normalised(fitted):
    """Transition rows, mixture weights and the initial distribution all sum to one."""
    assert_allclose(fitted.trans.sum(axis=1), np.ones(3), rtol=0, atol=1e-10)
    assert fitted.trans.min() >= 0.0
    assert fitted.init_priors.sum() == pytest.approx(1.0, abs=1e-10)
    assert fitted.init_priors.min() >= 0.0
    assert fitted.emissions.priors.sum() == pytest.approx(1.0, abs=1e-10)
    assert fitted.init_priors[0] > 0.99          # every sequence starts in the first cluster


def test_log_likelihood_increases_over_em_iterations():
    """Each expectation-maximisation step raises the likelihood of the sequences."""
    sequences = cluster_sequences(durations=(15, 85, 20), noise=0.35, n_sequences=3)
    likelihoods = []
    for n_iter in range(1, 6):
        model = HiddenMarkovModel.fit(sequences, 3, max_iter=n_iter, tol=0.0)
        likelihoods.append(np.mean([model._forward_backward(s)[2] for s in sequences]))
    assert np.all(np.diff(likelihoods) > 0.0)
    assert likelihoods[-1] - likelihoods[0] > 5.0


def test_state_posteriors_sum_to_one_at_every_step(fitted, clusters):
    """The state posteriors are a distribution over states at each timestep."""
    gamma = fitted.state_posteriors(clusters[0])
    assert gamma.shape == (3, len(clusters[0]))
    assert_allclose(gamma.sum(axis=0), np.ones(gamma.shape[1]), rtol=0, atol=1e-12)
    assert gamma.min() >= 0.0


def test_transition_posteriors_carry_unit_mass_at_every_step(fitted, clusters):
    """The transition posteriors sum to one per step and marginalise to the states."""
    gamma, zeta, _ = fitted._forward_backward(clusters[0])
    T = len(clusters[0])
    assert zeta.shape == (3, 3, T - 1)
    assert_allclose(zeta.sum(axis=(0, 1)), np.ones(T - 1), rtol=0, atol=1e-10)
    assert_allclose(zeta.sum(axis=1), gamma[:, :-1], rtol=0, atol=1e-10)


def test_fit_is_deterministic(clusters):
    """Two fits of the same sequences give bit-identical parameters."""
    first = HiddenMarkovModel.fit(clusters, 3)
    second = HiddenMarkovModel.fit(clusters, 3)
    assert np.array_equal(first.emissions.mu, second.emissions.mu)
    assert np.array_equal(first.emissions.sigma, second.emissions.sigma)
    assert np.array_equal(first.trans, second.trans)
    assert np.array_equal(first.init_priors, second.init_priors)


def test_fit_is_equivariant_under_a_translation(clusters):
    """Translating the data translates the means and leaves the rest unchanged."""
    offset = np.array([3.5, -2.25])
    plain = HiddenMarkovModel.fit(clusters, 3)
    shifted = HiddenMarkovModel.fit([s + offset for s in clusters], 3)
    assert_allclose(shifted.emissions.mu, plain.emissions.mu + offset, rtol=0, atol=1e-12)
    assert_allclose(shifted.emissions.sigma, plain.emissions.sigma, rtol=0, atol=1e-12)
    assert_allclose(shifted.trans, plain.trans, rtol=0, atol=1e-12)


def test_fit_is_not_invariant_under_a_scaling(clusters):
    """Scaling the data scales the covariances by the square of the factor."""
    scale = 7.0
    plain = HiddenMarkovModel.fit(clusters, 3, reg=1e-12)
    scaled = HiddenMarkovModel.fit([s * scale for s in clusters], 3, reg=1e-12)
    assert_allclose(scaled.emissions.sigma, scale ** 2 * plain.emissions.sigma,
                    rtol=1e-6, atol=1e-9)
    assert np.abs(scaled.emissions.sigma - plain.emissions.sigma).max() > 0.1


# The first and last state emit around the same point, so nothing in a single
# observation separates them and only the transitions say which one it came from.
DECODE_MEANS = np.array([[0.0, 0.0], [5.0, 0.0], [0.3, 0.0]])
DECODE_NOISE = 0.5
DECODE_TRANS = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])


def overlapping_chain():
    """A left-to-right chain that can never return to its first state."""
    emissions = GaussianMixture(DECODE_MEANS,
                                np.tile(DECODE_NOISE ** 2 * np.eye(2), (3, 1, 1)),
                                np.full(3, 1 / 3))
    return HiddenMarkovModel(emissions, DECODE_TRANS, np.array([1.0, 0.0, 0.0]))


def path_log_probability(model, obs, path):
    """Joint log probability of one state path and the observations along it."""
    tiny = np.finfo(float).tiny
    log_b = model.emissions.log_likelihoods(obs)
    return float(np.log(model.init_priors[path[0]] + tiny)
                 + np.log(model.trans[path[:-1], path[1:]] + tiny).sum()
                 + log_b[path, np.arange(len(path))].sum())


@pytest.mark.parametrize("durations", [(6, 7, 5), (4, 4, 10)])
def test_viterbi_decodes_the_path_that_a_nearest_mean_assignment_gets_wrong(durations):
    """Viterbi returns the most likely path, which the nearest emission mean misreads."""
    model = overlapping_chain()
    path = np.repeat([0, 1, 2], durations)
    obs = DECODE_MEANS[path] + DECODE_NOISE * np.random.default_rng(3).normal(
        size=(len(path), 2))
    decoded = model.viterbi(obs)
    nearest = np.linalg.norm(obs[:, None, :] - DECODE_MEANS[None, :, :],
                             axis=2).argmin(axis=1)
    misread = nearest != path

    assert decoded.shape == (len(path),)
    assert np.issubdtype(decoded.dtype, np.integer)
    assert misread.sum() >= 3
    assert np.array_equal(decoded, path)
    assert np.array_equal(decoded != nearest, misread)
    elsewhere = [path_log_probability(model, obs,
                                      np.where(np.arange(len(path)) == t, state, decoded))
                 for t in range(len(path)) for state in range(3) if state != decoded[t]]
    assert path_log_probability(model, obs, decoded) > max(elsewhere) + 10.0


def test_sequence_too_short_to_fill_the_time_bins_is_rejected():
    """A sequence that cannot fill every time bin is an error, not a model of NaNs."""
    short = np.random.default_rng(0).normal(size=(2, 2))
    with pytest.raises(ValueError, match="time bin"):
        HiddenMarkovModel.fit([short], 3)
