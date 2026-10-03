"""Test estimation for the TPAG replication package."""

import numpy as np
from scipy.stats import beta

from tpag.estimation import (
    EstimationModel,
    clopper_pearson_lower,
    clopper_pearson_upper,
)


def test_upper_matches_beta_quantile():
    k, m, eta = 3, 100, 0.05
    assert np.isclose(clopper_pearson_upper(k, m, eta), beta.ppf(1 - eta, k + 1, m - k))


def test_lower_matches_beta_quantile():
    k, m, eta = 7, 100, 0.05
    assert np.isclose(clopper_pearson_lower(k, m, eta), beta.ppf(eta, k, m - k + 1))


def test_edge_cases():
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0  # no data -> sound upper bound is 1
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0  # all violations
    assert clopper_pearson_lower(0, 100, 0.05) == 0.0  # no violations -> lower 0
    assert clopper_pearson_lower(0, 0, 0.05) == 0.0


def test_upper_ge_point_ge_lower():
    k, m, eta = 5, 50, 0.05
    up = clopper_pearson_upper(k, m, eta)
    lo = clopper_pearson_lower(k, m, eta)
    assert lo <= k / m <= up


def test_upper_limit_coverage():
    """For a fixed true rate p, the one-sided upper limit covers p at least
    (1 - eta) of the time across repeated experiments."""
    rng = np.random.default_rng(0)
    p, m, eta, trials = 0.1, 80, 0.05, 4000
    covered = 0
    for _ in range(trials):
        k = int(rng.binomial(m, p))
        if clopper_pearson_upper(k, m, eta) >= p:
            covered += 1
    # exact CP is conservative; empirical coverage should comfortably exceed 1-eta
    assert covered / trials >= 1 - eta


def test_estimation_model_confidence_accounting():
    est = EstimationModel()
    for node in ("a", "b", "c"):
        c = est.marginal(node)
        for _ in range(100):
            c.observe(False)
    est.joint(frozenset({"a", "b"}))
    # N = 3 marginals + 1 joint = 4 estimated quantities
    assert est.num_estimated_quantities() == 4
    assert np.isclose(est.overall_confidence(0.01), 1 - 4 * 0.01)
