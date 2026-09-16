"""Tests for the Gaussian mixture and the operations the model is built from.

Everything here is checked against a closed form or an independent computation:
the affine map is recomputed component by component, the product of Gaussians is
compared with the information-form solution, and the densities are compared with
``scipy.stats.multivariate_normal``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal

from tplqt.gaussian import GaussianMixture

SKEW = np.array([[2.0, 1.0, 0.0],
                 [0.0, 3.0, 1.0],
                 [1.0, 0.0, 1.0]])          # non-orthogonal and non-symmetric


def mixture(n_states: int, n_dim: int, seed: int = 0, scale: float = 1.0):
    """A mixture with well-conditioned random covariances."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n_states, n_dim))
    root = rng.normal(size=(n_states, n_dim, n_dim))
    sigma = scale * (root @ root.transpose(0, 2, 1) + n_dim * np.eye(n_dim))
    return GaussianMixture(mu, sigma)


@pytest.mark.parametrize("mu_shape, sigma_shape", [
    ((3, 4), (3, 4, 5)),        # covariance not square
    ((3, 4), (2, 4, 4)),        # wrong number of states
    ((3, 4), (3, 3, 3)),        # wrong dimension
    ((3, 4), (4, 4)),           # covariance missing the state axis
    ((4,), (4, 4, 4)),          # mean missing the state axis
])
def test_mismatched_mean_and_covariance_shapes_are_rejected(mu_shape, sigma_shape):
    """Shapes that do not describe K Gaussians over R^D raise ValueError."""
    with pytest.raises(ValueError):
        GaussianMixture(np.zeros(mu_shape), np.zeros(sigma_shape))


def test_priors_default_to_uniform():
    """Omitted mixing weights are uniform and sum to one."""
    gmm = GaussianMixture(np.zeros((5, 2)), np.broadcast_to(np.eye(2), (5, 2, 2)))
    np.testing.assert_allclose(gmm.priors, np.full(5, 0.2), rtol=0, atol=1e-15)
    assert gmm.priors.sum() == pytest.approx(1.0, abs=1e-15)


def test_precision_inverts_the_covariance_and_is_cached():
    """The precision is the matrix inverse of the covariance, computed once."""
    gmm = mixture(3, 4, seed=1)
    identity = np.einsum("kij,kjl->kil", gmm.precision, gmm.sigma)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(4), (3, 4, 4)),
                               rtol=0, atol=1e-10)
    assert gmm.precision is gmm.precision


def test_marginal_keeps_the_entries_of_the_selected_block():
    """Marginal means and covariances are the corresponding full entries."""
    gmm = mixture(3, 5, seed=2)
    gmm.priors = np.array([0.2, 0.5, 0.3])
    block = slice(1, 4)
    marginal = gmm.marginal(block)
    np.testing.assert_array_equal(marginal.mu, gmm.mu[:, 1:4])
    np.testing.assert_array_equal(marginal.sigma, gmm.sigma[:, 1:4, 1:4])
    np.testing.assert_array_equal(marginal.priors, gmm.priors)


def test_marginal_splits_a_fitted_mixture_into_its_per_frame_blocks(model):
    """The per-frame blocks of a fitted mixture concatenate back to the whole."""
    emissions = model.hmm.emissions
    n_state = model.n_state
    assert emissions.n_dim == n_state * len(model.frames)
    blocks = [emissions.marginal(slice(i * n_state, (i + 1) * n_state))
              for i in range(len(model.frames))]
    assert [b.mu.shape for b in blocks] == [(model.n_states, n_state)] * len(model.frames)
    np.testing.assert_array_equal(np.concatenate([b.mu for b in blocks], axis=1),
                                  emissions.mu)


def test_transform_matches_an_independent_affine_computation():
    """A non-orthogonal map sends mu to A mu + b and sigma to A sigma A^T."""
    gmm = mixture(3, 3, seed=3)
    b = np.array([0.5, -1.25, 2.0])
    moved = gmm.transform(SKEW, b)
    for k in range(gmm.n_states):
        np.testing.assert_allclose(moved.mu[k], SKEW @ gmm.mu[k] + b, rtol=1e-12, atol=0)
        np.testing.assert_allclose(moved.sigma[k], SKEW @ gmm.sigma[k] @ SKEW.T,
                                   rtol=1e-12, atol=0)
        # negative control: the transposed map is a different covariance here
        assert not np.allclose(moved.sigma[k], SKEW.T @ gmm.sigma[k] @ SKEW, atol=1e-6)


def test_rotation_leaves_the_covariance_eigenvalues_unchanged():
    """A rotation moves the means but preserves the covariance spectrum."""
    gmm = mixture(3, 3, seed=4)
    A = R.from_rotvec([0.3, -0.7, 1.1]).as_matrix()
    turned = gmm.transform(A, np.zeros(3))
    np.testing.assert_allclose(np.linalg.eigvalsh(turned.sigma),
                               np.linalg.eigvalsh(gmm.sigma), rtol=1e-10, atol=0)
    np.testing.assert_allclose(turned.mu, gmm.mu @ A.T, rtol=1e-12, atol=0)


def test_product_of_one_dimensional_gaussians_matches_the_closed_form():
    """In one dimension precisions add and the mean is their weighted average."""
    mu1, var1 = 1.5, 0.25
    mu2, var2 = -0.5, 4.0
    a = GaussianMixture([[mu1]], [[[var1]]])
    b = GaussianMixture([[mu2]], [[[var2]]])
    prod = a * b
    expected_var = 1.0 / (1.0 / var1 + 1.0 / var2)
    expected_mu = expected_var * (mu1 / var1 + mu2 / var2)
    assert prod.sigma[0, 0, 0] == pytest.approx(expected_var, rel=1e-12)
    assert prod.mu[0, 0] == pytest.approx(expected_mu, rel=1e-12)
    assert prod.precision[0, 0, 0] == pytest.approx(1.0 / var1 + 1.0 / var2, rel=1e-12)


def test_product_precision_adds_and_the_mean_solves_the_information_system():
    """The product has precision P1 + P2 and mean solving P mu = P1 mu1 + P2 mu2."""
    a, b = mixture(3, 4, seed=5), mixture(3, 4, seed=6)
    prod = a * b
    np.testing.assert_allclose(prod.precision, a.precision + b.precision,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(prod.precision, np.linalg.inv(prod.sigma),
                               rtol=1e-8, atol=1e-10)
    left = np.einsum("kij,kj->ki", prod.precision, prod.mu)
    right = (np.einsum("kij,kj->ki", a.precision, a.mu)
             + np.einsum("kij,kj->ki", b.precision, b.mu))
    np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-12)


def test_product_is_commutative_up_to_the_priors():
    """Swapping the operands leaves mu and precision alone and takes the left priors."""
    a, b = mixture(3, 4, seed=7), mixture(3, 4, seed=8)
    a.priors = np.array([0.6, 0.3, 0.1])
    b.priors = np.array([0.1, 0.2, 0.7])
    forward, backward = a * b, b * a
    np.testing.assert_allclose(forward.mu, backward.mu, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(forward.precision, backward.precision, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(forward.priors, a.priors)
    np.testing.assert_array_equal(backward.priors, b.priors)


def test_product_is_pulled_towards_the_more_certain_mixture():
    """With isotropic covariances the product sits at the analytic variance ratio."""
    var_loose, var_tight = 1.0, 1e-6
    loose = GaussianMixture(np.zeros((2, 3)),
                            var_loose * np.broadcast_to(np.eye(3), (2, 3, 3)))
    tight = GaussianMixture(np.ones((2, 3)),
                            var_tight * np.broadcast_to(np.eye(3), (2, 3, 3)))
    prod = loose * tight
    pull = (np.linalg.norm(prod.mu - tight.mu, axis=1)
            / np.linalg.norm(loose.mu - tight.mu, axis=1))
    np.testing.assert_allclose(pull, var_tight / (var_loose + var_tight),
                               rtol=1e-9, atol=0)


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_multiplying_mixtures_of_different_shape_is_rejected(shape):
    """A different number of states or dimensions raises ValueError."""
    with pytest.raises(ValueError):
        mixture(3, 4, seed=9) * mixture(*shape, seed=10)


@pytest.mark.parametrize("n_points", [1, 7])
def test_log_likelihoods_match_the_multivariate_normal_density(n_points):
    """Component log densities equal the reference multivariate normal log pdf."""
    gmm = mixture(3, 4, seed=11)
    x = np.random.default_rng(12).normal(size=(n_points, 4))
    values = gmm.log_likelihoods(x)
    assert values.shape == (3, n_points)
    for k in range(gmm.n_states):
        reference = multivariate_normal(mean=gmm.mu[k], cov=gmm.sigma[k]).logpdf(x)
        np.testing.assert_allclose(values[k], reference, rtol=1e-10, atol=1e-10)


def test_log_likelihoods_accept_a_single_point_as_one_row():
    """A one-dimensional input is read as a single observation."""
    gmm = mixture(3, 4, seed=13)
    x = np.array([0.2, -0.4, 1.0, 0.7])
    values = gmm.log_likelihoods(x)
    assert values.shape == (3, 1)
    np.testing.assert_allclose(values, gmm.log_likelihoods(x[None, :]), rtol=0, atol=0)


def test_sequence_returns_the_components_in_the_given_order():
    """A state sequence is expanded into per-timestep means and covariances."""
    gmm = mixture(4, 3, seed=14)
    states = [2, 0, 0, 3, 1]
    mu_seq, sigma_seq = gmm.sequence(states)
    assert mu_seq.shape == (5, 3) and sigma_seq.shape == (5, 3, 3)
    np.testing.assert_array_equal(mu_seq, gmm.mu[states])
    np.testing.assert_array_equal(sigma_seq, gmm.sigma[states])
