"""Structure for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product


def minimal_sets(family: list[frozenset[str]]) -> list[frozenset[str]]:
    """Keep only the (subset-)minimal members of a family of sets."""
    uniq = set(family)
    out: list[frozenset[str]] = []
    for s in uniq:
        if not any(other < s for other in uniq if other != s):
            out.append(s)
    # deterministic ordering for reproducible bounds/reports
    return sorted(out, key=lambda s: (len(s), sorted(s)))


def coherence_violation(structure: Structure) -> tuple[frozenset[str], str] | None:
    """Check the *coherence* (monotonicity) hypothesis of Assumption 2 explicitly.

    The composition theorem (and the cut-set cover, Lemma 2) require the structure
    function ``phi`` of ``not Phi`` to be **monotone**: adding violations never
    removes a system violation (``x <= y => phi(x) <= phi(y)``). For a structure
    expressed by minimal cut sets this holds *by construction*, but a hand-declared
    structure (or a future :class:`Structure` subclass whose ``failed`` disagrees
    with its ``cut_sets``) can violate it. We verify the upward-closure condition --
    the exact coherence property -- on the witnesses that can break it:

    * each minimal cut ``kappa`` (must be failed),
    * each ``kappa`` with one member dropped (the minimality witness), and
    * for every failed probe ``W`` and atom ``a not in W``, ``W u {a}`` is still
      failed (monotonicity);

    plus ``phi(emptyset)`` false and ``phi(all)`` true (non-triviality). Returns the
    ``(witness_set, reason)`` of the first violation found, or ``None`` if coherent.

    Note: this checks the *propositional* coherence of the declared failure
    structure. Semantic non-monotonicity -- e.g. a guarantee whose *violation*
    triggers a compensating action that prevents ``not Phi`` -- cannot be expressed
    in the cut-set algebra at all and is out of scope (the manuscript excludes such
    non-coherent compositions).
    """
    cuts = structure.cut_sets()
    atoms = sorted(structure.atoms())
    if not atoms:
        return None
    full = set(atoms)

    # Non-triviality: satisfying all guarantees is safe; violating all is unsafe.
    if structure.failed(set()):
        return frozenset(), "phi(no violations) is True (not a coherent failure structure)"
    if cuts and not structure.failed(full):
        return frozenset(full), "phi(all violations) is False (semantic/cut-set mismatch)"

    # Build the probe sets whose upward closure must hold.
    probes: list[set[str]] = [set(c) for c in cuts]
    for c in cuts:
        for m in c:  # one-member-dropped witnesses of minimality
            probes.append(set(c) - {m})
    seen: set[frozenset[str]] = set()
    for W in probes:
        key = frozenset(W)
        if key in seen:
            continue
        seen.add(key)
        if not structure.failed(W):
            continue
        # W is failed -> every superset must remain failed (monotonicity).
        for a in atoms:
            if a in W:
                continue
            if not structure.failed(W | {a}):
                return key, (
                    f"adding violation {a!r} to {sorted(W)} removes the system "
                    "violation: failure structure is non-monotone (incoherent)"
                )
    return None


class Structure:
    """Abstract node of a failure-structure expression."""

    def cut_sets(self) -> list[frozenset[str]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def atoms(self) -> set[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def failed(self, violated: set[str]) -> bool:
        """Semantic ``phi(b)``: True iff the violated set contains some cut."""
        return any(cut <= set(violated) for cut in self.cut_sets())


@dataclass
class Atom(Structure):
    component: str

    def cut_sets(self) -> list[frozenset[str]]:
        return [frozenset({self.component})]

    def atoms(self) -> set[str]:
        return {self.component}


@dataclass
class Series(Structure):
    """OR of children: the system fails if any child fails (delegation chain)."""

    children: list[Structure]

    def cut_sets(self) -> list[frozenset[str]]:
        fam: list[frozenset[str]] = []
        for ch in self.children:
            fam.extend(ch.cut_sets())
        return minimal_sets(fam)

    def atoms(self) -> set[str]:
        out: set[str] = set()
        for ch in self.children:
            out |= ch.atoms()
        return out


@dataclass
class Parallel(Structure):
    """AND of children: the system fails only if every child fails (redundancy)."""

    children: list[Structure]

    def cut_sets(self) -> list[frozenset[str]]:
        child_cuts = [ch.cut_sets() for ch in self.children]
        fam: list[frozenset[str]] = []
        for combo in product(*child_cuts):
            union: frozenset[str] = frozenset().union(*combo)
            fam.append(union)
        return minimal_sets(fam)

    def atoms(self) -> set[str]:
        out: set[str] = set()
        for ch in self.children:
            out |= ch.atoms()
        return out


@dataclass
class KofN(Structure):
    """``k``-out-of-``n``: works iff >= k children work; fails iff > n-k fail."""

    k: int
    children: list[Structure]

    def cut_sets(self) -> list[frozenset[str]]:
        n = len(self.children)
        fail_count = n - self.k + 1  # number of child-failures that fails the system
        if fail_count <= 0:
            return []
        fam: list[frozenset[str]] = []
        for subset in combinations(range(n), fail_count):
            child_cuts = [self.children[i].cut_sets() for i in subset]
            for combo in product(*child_cuts):
                fam.append(frozenset().union(*combo))
        return minimal_sets(fam)

    def atoms(self) -> set[str]:
        out: set[str] = set()
        for ch in self.children:
            out |= ch.atoms()
        return out


@dataclass
class ExplicitCutSets(Structure):
    """A structure declared directly by its minimal cut sets."""

    cuts: list[frozenset[str]]

    def cut_sets(self) -> list[frozenset[str]]:
        return minimal_sets([frozenset(c) for c in self.cuts])

    def atoms(self) -> set[str]:
        out: set[str] = set()
        for c in self.cuts:
            out |= set(c)
        return out


# -- convenience builders ---------------------------------------------------
def series(*items) -> Series:
    return Series([_lift(x) for x in items])


def parallel(*items) -> Parallel:
    return Parallel([_lift(x) for x in items])


def redundancy(*items) -> Parallel:
    """Alias for ``parallel``: defense-in-depth where all safeguards must miss."""
    return Parallel([_lift(x) for x in items])


def kofn(k: int, *items) -> KofN:
    return KofN(k, [_lift(x) for x in items])


def _lift(x) -> Structure:
    return x if isinstance(x, Structure) else Atom(str(x))


# -- adequacy obligation (Assumption 1) -------------------------------------
@dataclass
class AdequacyReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    cut_sets: list[frozenset[str]]

    def __bool__(self) -> bool:
        return self.ok


def check_adequacy(structure: Structure, declared_components: set[str]) -> AdequacyReport:
    """Propositional/structural check of ``A_env and AND_i G_i => Phi`` (Assm. 1).

    Verifies the declared failure structure is well-formed and adequate:

    * every atom of the structure is a declared component guarantee;
    * the structure is *coherent* (Assumption 2) -- at least one cut, no empty cut,
      and the structure function is **monotone** (adding violations never removes a
      system violation), checked explicitly by :func:`coherence_violation`, so that
      satisfying all guarantees (empty violation set) implies ``Phi``;
    * every declared component appears in some cut (else its guarantee is
      irrelevant to ``Phi`` -- a warning, not an error).

    General automata-level entailment over arbitrary regular predicates is a
    documented extension point; this covers the propositional fragment the cut-set
    bound (Thm. 1) actually consumes.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cuts = structure.cut_sets()
    atoms = structure.atoms()

    unknown = atoms - declared_components
    if unknown:
        errors.append(f"structure references undeclared components: {sorted(unknown)}")

    if not cuts:
        errors.append("structure has no cut sets: not Phi is unsatisfiable; check the spec")
    if any(len(c) == 0 for c in cuts):
        errors.append("structure has an empty cut set: Phi fails with no violation (incoherent)")

    # phi(emptyset) must be False: satisfying all guarantees implies Phi.
    if structure.failed(set()):
        errors.append("phi(no violations) is True: guarantees do not entail Phi (adequacy fails)")

    # Coherence/monotonicity (Assumption 2): reject non-monotone failure structures,
    # which the cut-set cover (Lemma 2) and Theorem 1 require but do not imply.
    viol = coherence_violation(structure)
    if viol is not None:
        witness, reason = viol
        if not (witness == frozenset() and structure.failed(set())):
            # avoid double-reporting the phi(emptyset) case already flagged above
            errors.append(f"coherence (Assumption 2) fails: {reason}")

    covered = set().union(*cuts) if cuts else set()
    irrelevant = declared_components - covered
    if irrelevant:
        warnings.append(
            f"guarantees not contributing to any cut of Phi (dead weight): {sorted(irrelevant)}"
        )

    return AdequacyReport(ok=not errors, errors=errors, warnings=warnings, cut_sets=cuts)
