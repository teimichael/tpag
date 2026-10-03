"""Test substitutability for the TPAG replication package."""

import numpy as np

from tpag.certificate import certify_exact
from tpag.contracts import Contract
from tpag.predicates import always, guard_never_wrongly_approves
from tpag.structure import parallel, series


def _guard(name, eps):
    return Contract(
        always(lambda e: True, "true"), guard_never_wrongly_approves(name=f"{name}_G"), eps, name
    )


def test_refining_contract_has_smaller_epsilon():
    g = always(lambda e: True, "true")
    G = guard_never_wrongly_approves(name="shared_G")
    old = Contract(g, G, 0.10, "g1")
    new = Contract(g, G, 0.04, "g1")  # same A, G; smaller epsilon
    assert new.refines(old)
    assert not old.refines(new)


def test_bound_non_increasing_under_refinement_redundancy():
    structure = parallel("g1", "g2")
    joints = {frozenset({"g1", "g2"}): 0.06}
    before = certify_exact(structure, {"g1": 0.10, "g2": 0.10}, pair_joints=joints).bound
    after = certify_exact(structure, {"g1": 0.04, "g2": 0.10}, pair_joints=joints).bound
    assert after <= before + 1e-12


def test_bound_non_increasing_under_refinement_series():
    structure = series("g1", "g2", "g3")
    before = certify_exact(structure, {"g1": 0.05, "g2": 0.05, "g3": 0.05}).bound
    after = certify_exact(structure, {"g1": 0.01, "g2": 0.05, "g3": 0.05}).bound
    assert after <= before + 1e-12
    assert np.isclose(before, 0.15)
    assert np.isclose(after, 0.11)


def test_monotonicity_each_coordinate():
    structure = series("a", parallel("b", "c"))
    base = {"a": 0.05, "b": 0.2, "c": 0.2}
    b0 = certify_exact(structure, base).bound
    for node in base:
        tighter = dict(base)
        tighter[node] = base[node] * 0.5
        assert certify_exact(structure, tighter).bound <= b0 + 1e-12
