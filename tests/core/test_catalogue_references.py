"""Exact posteriorDB catalogue coordinates and an independent noncentered density."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import halfcauchy, norm

from corpora.posteriordb.catalogue.regenerate import (
    density_offset,
    verified_digest,
    verify_points,
)

CATALOGUE = Path(__file__).resolve().parents[2] / "corpora/posteriordb/catalogue"


def test_catalogue_coordinates_and_gradients():
    query = (CATALOGUE / "query.flatppl").read_bytes()
    fixtures = sorted(CATALOGUE.glob("*.json"))
    assert len(fixtures) == 24
    for path in fixtures:
        case = json.loads(path.read_text())
        p = case["provenance"]
        points = case["points"]
        assert len(points) == 3
        verify_points(points, p["point_hashes"])
        verified_digest(query, p["query_sha256"])
        offset = density_offset(p, points[0])
        assert offset == p["log_density_offset"]
        np.testing.assert_array_equal(
            np.array(p["stan_log_density"]) + offset, case["expected"]
        )
        np.testing.assert_allclose(
            p["stan_log_density_with_gradient"],
            p["stan_log_density"],
            atol=1e-8,
            rtol=1e-8,
        )
        for point, grad in zip(points, case["expected_grad"], strict=True):
            assert grad.keys() == point.keys()
            for name, value in grad.items():
                assert np.shape(value) == np.shape(point[name])
                assert np.all(np.isfinite(value))


def test_noncentered_eight_schools():
    case = json.loads(
        (CATALOGUE / "eight_schools-eight_schools_noncentered.json").read_text()
    )
    data = json.loads(
        (
            CATALOGUE.parent / "eight_schools_centered/data/eight_schools.json"
        ).read_text()
    )
    y, sigma = np.array(data["y"]), np.array(data["sigma"])
    for point, score, grad in zip(
        case["points"], case["expected"], case["expected_grad"], strict=True
    ):
        mu, tau = point["mu"], point["tau"]
        z = np.array(point["theta_trans"])
        theta = mu + tau * z
        score_analytic = (
            norm.logpdf(mu, scale=5)
            + halfcauchy.logpdf(tau, scale=5)
            + norm.logpdf(z).sum()
            + norm.logpdf(y, theta, sigma).sum()
        )
        np.testing.assert_allclose(score, score_analytic, rtol=0, atol=2e-14)
        dtheta = (y - theta) / sigma**2
        np.testing.assert_allclose(
            grad["theta_trans"], -z + tau * dtheta, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            grad["mu"], dtheta.sum() - mu / 25, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            grad["tau"], z @ dtheta - 2 * tau / (25 + tau**2), rtol=0, atol=1e-15
        )
