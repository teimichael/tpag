"""Test bound specializations for the TPAG replication package."""

import numpy as np

from tpag.bound import compute_bound
from tpag.structure import parallel, series


def test_series_reduces_to_boole_without_joints():
    s = series("a", "b", "c")
    eps = {"a": 0.01, "b": 0.02, "c": 0.03}
    res = compute_bound(s.cut_sets(), eps)
    assert np.isclose(res.b_boole, 0.06)  # sum of marginals (Boole)
    assert np.isclose(res.b_hw, 0.06)  # no cut joints supplied -> no tightening


def test_hunter_worsley_tightens_under_positive_correlation():
    s = series("a", "b")
    eps = {"a": 0.1, "b": 0.1}
    # measured positive co-violation between the two singleton cuts {a},{b}
    joint = 0.05

    def cut_joint_lower(ca, cb):
        return joint if (ca | cb) == frozenset({"a", "b"}) else 0.0

    res = compute_bound(s.cut_sets(), eps, cut_joint_lower=cut_joint_lower)
    assert np.isclose(res.b_boole, 0.2)
    assert np.isclose(res.b_hw, 0.2 - 0.05)  # Boole minus the spanning-tree joint
    assert res.b_hw < res.b_boole


def test_redundancy_reduces_to_min_marginal():
    s = parallel("g1", "g2", "g3")
    eps = {"g1": 0.2, "g2": 0.05, "g3": 0.3}
    res = compute_bound(s.cut_sets(), eps)
    # single cut {g1,g2,g3}; Fréchet upper bound = min marginal
    assert np.isclose(res.b_boole, 0.05)
    assert np.isclose(res.b_hw, 0.05)


def test_redundancy_tightened_by_measured_pair():
    s = parallel("g1", "g2")
    eps = {"g1": 0.2, "g2": 0.2}
    pair = {frozenset({"g1", "g2"}): 0.04}  # measured co-violation < min marginal
    res = compute_bound(s.cut_sets(), eps, pair_upper=pair)
    assert np.isclose(res.b_boole, 0.04)  # tightened from min(0.2,0.2)=0.2 to 0.04


def test_hw_never_looser_than_boole():
    s = series("a", "b", "c", "d")
    eps = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}

    def cjl(ca, cb):
        return 0.03

    res = compute_bound(s.cut_sets(), eps, cut_joint_lower=cjl)
    assert res.b_hw <= res.b_boole
