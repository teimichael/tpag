"""Contracts for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .predicates import Predicate

# An entailment oracle decides whether predicate `p` implies predicate `q`
# (every trace satisfying p also satisfies q). Soundness of refinement depends on
# this oracle; the default below is deliberately conservative.
Entailment = Callable[[Predicate, Predicate], bool]


def identity_entailment(p: Predicate, q: Predicate) -> bool:
    """Conservative default: entailment holds only for the same predicate object
    or two predicates with identical name+kind. Sound (never claims a false
    implication) but incomplete. Replace with a stronger oracle (e.g. automata
    inclusion) to certify non-trivial refinements."""
    return p is q or (p.name == q.name and p.kind == q.kind)


@dataclass(frozen=True)
class Contract:
    """A probabilistic assume-guarantee contract for one component."""

    assumption: Predicate
    guarantee: Predicate
    epsilon: float
    name: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon < 1.0):
            raise ValueError(f"epsilon must be in [0,1), got {self.epsilon}")

    def refines(self, other: Contract, entailment: Entailment = identity_entailment) -> bool:
        """Return True iff ``self <= other`` (self refines other; Def. 4).

        Conditions: ``other.A => self.A`` (weaker assumption), ``self.G =>
        other.G`` (stronger guarantee), ``self.epsilon <= other.epsilon``.
        """
        weaker_assumption = entailment(other.assumption, self.assumption)
        stronger_guarantee = entailment(self.guarantee, other.guarantee)
        smaller_eps = self.epsilon <= other.epsilon + 1e-12
        return weaker_assumption and stronger_guarantee and smaller_eps


def is_preorder_reflexive(c: Contract, entailment: Entailment = identity_entailment) -> bool:
    return c.refines(c, entailment)


def refinement_transitive(
    a: Contract, b: Contract, c: Contract, entailment: Entailment = identity_entailment
) -> bool:
    """If ``a <= b`` and ``b <= c`` then ``a <= c`` (preorder transitivity check)."""
    if a.refines(b, entailment) and b.refines(c, entailment):
        return a.refines(c, entailment)
    return True  # vacuously holds when the antecedent is false


def local_substitution_sound(
    new: Contract, old: Contract, entailment: Entailment = identity_entailment
) -> bool:
    """Lemma 1 (local substitution): if ``new <= old`` then any component
    satisfying ``new`` also satisfies ``old``. We expose the *checkable*
    precondition (the refinement) that licenses the substitution."""
    return new.refines(old, entailment)
