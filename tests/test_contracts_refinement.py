"""Test contracts refinement for the TPAG replication package."""

import pytest

from tpag.contracts import Contract, local_substitution_sound
from tpag.predicates import always, no_side_effect_without_label


def _eps_contract(eps: float, name="c") -> Contract:
    g = no_side_effect_without_label("approved", name=f"{name}_G")
    a = always(lambda e: True, name=f"{name}_A")
    return Contract(assumption=a, guarantee=g, epsilon=eps, name=name)


def test_epsilon_bounds_validated():
    with pytest.raises(ValueError):
        Contract(_eps_contract(0.1).assumption, _eps_contract(0.1).guarantee, 1.0)


def test_refinement_reflexive():
    c = _eps_contract(0.1)
    assert c.refines(c)


def test_refinement_smaller_epsilon():
    # Same A,G; smaller epsilon refines larger epsilon.
    strong = _eps_contract(0.01, "c")
    weak = _eps_contract(0.10, "c")
    # Reuse identical predicate objects so the identity oracle accepts A=>A', G'=>G.
    strong = Contract(weak.assumption, weak.guarantee, 0.01, "c")
    assert strong.refines(weak)
    assert not weak.refines(strong)  # 0.10 <= 0.01 is false


def test_local_substitution_precondition():
    weak = _eps_contract(0.10, "c")
    strong = Contract(weak.assumption, weak.guarantee, 0.02, "c")
    assert local_substitution_sound(strong, weak)


def test_identity_oracle_is_conservative():
    # Different guarantee predicates are NOT assumed to entail each other.
    c1 = _eps_contract(0.01, "a")
    c2 = _eps_contract(0.01, "b")
    assert not c1.refines(c2)  # names differ -> conservative oracle rejects
