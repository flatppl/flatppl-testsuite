"""Sampler validation checks support and shape independently of matching moments."""

import numpy as np

from flatppl_testsuite.unified.sample_checks import check_fanout_distribution


def test_dirichlet_fanout_requires_nonnegative_coordinates():
    alpha = np.array([2.0, 3.0, 5.0])
    mean = alpha / alpha.sum()
    covariance = (np.diag(mean) - np.outer(mean, mean)) / (alpha.sum() + 1)
    values, vectors = np.linalg.eigh(covariance)
    directions = vectors[:, 1:] * np.sqrt(values[1:])
    # Four symmetric atoms, each with probability 1/400, preserve the target
    # covariance. Their sum stays one, but their coordinates leave the simplex.
    points = np.concatenate([
        mean[None, :], mean + np.sqrt(200) * directions.T,
        mean - np.sqrt(200) * directions.T,
    ])
    invalid = np.repeat(points, [396, 1, 1, 1, 1], axis=0)
    valid = np.random.default_rng(27).dirichlet(alpha, 10000)
    recipe, params = {"fanout_simplex": True}, {"alpha": alpha}

    assert check_fanout_distribution("dirichlet", valid, recipe, params).status == "passed"
    assert check_fanout_distribution("dirichlet", invalid, recipe, params).status == "failed"


def test_dirichlet_fanout_rejects_atoms_with_correct_support_and_moments():
    alpha = np.array([2.0, 3.0, 5.0])
    mean = alpha / alpha.sum()
    covariance = (np.diag(mean) - np.outer(mean, mean)) / (alpha.sum() + 1)
    values, vectors = np.linalg.eigh(covariance)
    directions = vectors[:, 1:] * np.sqrt(values[1:])
    # Equal mass at four interior simplex points gives exactly the Dirichlet
    # mean and covariance, but each marginal has four atoms, not a Beta law.
    points = np.concatenate([
        mean + np.sqrt(2) * directions.T,
        mean - np.sqrt(2) * directions.T,
    ])
    samples = np.repeat(points, 2500, axis=0)
    assert np.all((points > 0) & (points < 1))
    np.testing.assert_allclose(points.sum(axis=1), 1)
    np.testing.assert_allclose(points.mean(axis=0), mean)
    np.testing.assert_allclose(np.cov(points, rowvar=False, ddof=0), covariance)

    result = check_fanout_distribution(
        "dirichlet", samples, {"fanout_simplex": True}, {"alpha": alpha},
    )
    assert result.status == "failed"
