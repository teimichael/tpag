"""Bound for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx


@dataclass
class BoundResult:
    b_boole: float
    b_hw: float
    per_cut: list[tuple[frozenset[str], float]]
    delta_hw: float
    hw_tree_edges: list[tuple[int, int]]
    regime: str  # "series" | "series-parallel" | "general"

    def best(self) -> float:
        return self.b_hw

    def as_dict(self) -> dict:
        return {
            "b_boole": self.b_boole,
            "b_hw": self.b_hw,
            "delta_hw": self.delta_hw,
            "regime": self.regime,
            "per_cut": [{"cut": sorted(c), "u": u} for c, u in self.per_cut],
        }


def _cut_upper(
    cut: frozenset[str],
    eps_upper: dict[str, float],
    pair_upper: dict[frozenset[str], float] | None,
) -> float:
    """u_kappa = min( min_i eps_i , min_{i!=j} kappa_ij , kappa_cut ) -- the Fréchet
    upper bound on Pr[∩_{i in cut} B_i], tightened by any measured pairwise joint and
    by a measured joint over the *whole* cut (a measured upper bound on Pr[∩ B_i] is
    itself a valid upper bound on the cut, for any cut size; this is what makes the
    k>=3 bound use the measured k-way co-violation rather than degrading to the
    min-marginal)."""
    if not cut:
        return 0.0
    u = min(eps_upper.get(i, 1.0) for i in cut)
    if pair_upper and len(cut) >= 2:
        members = sorted(cut)
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                key = frozenset({members[a_idx], members[b_idx]})
                if key in pair_upper:
                    u = min(u, pair_upper[key])
        # a measured joint over the entire cut (size k) tightens directly (k>=3)
        if cut in pair_upper:
            u = min(u, pair_upper[cut])
    return min(1.0, max(0.0, u))


def compute_bound(
    cut_sets: list[frozenset[str]],
    eps_upper: dict[str, float],
    pair_upper: dict[frozenset[str], float] | None = None,
    cut_joint_lower: Callable[[frozenset[str], frozenset[str]], float] | None = None,
    regime: str = "general",
) -> BoundResult:
    """Compute ``B_Boole`` and ``B_HW`` for a failure structure.

    Parameters
    ----------
    cut_sets:
        Minimal cut sets of ``not Phi`` (from :mod:`tpag.structure`).
    eps_upper:
        Upper bound (or exact value) on each component's ``Pr[B_i]``.
    pair_upper:
        Optional upper bounds on pairwise joints ``Pr[B_i ∩ B_j]`` to tighten
        multi-component cuts via Fréchet.
    cut_joint_lower:
        Optional ``(cut_a, cut_b) -> ρ`` giving a *lower* bound on
        ``Pr[D_a ∩ D_b]`` for the Hunter--Worsley correction. For singleton
        (series) cuts this is exactly a lower bound on ``Pr[B_i ∩ B_j]``. If
        omitted, ``Δ_HW = 0`` and ``B_HW = B_Boole`` (still sound, just not
        tightened).
    regime:
        Topology tag for reporting ("series" / "series-parallel" / "general").
    """
    per_cut = [(cut, _cut_upper(cut, eps_upper, pair_upper)) for cut in cut_sets]
    raw_sum = sum(u for _, u in per_cut)
    b_boole = min(1.0, raw_sum)

    delta_hw = 0.0
    tree_edges: list[tuple[int, int]] = []
    if cut_joint_lower is not None and len(cut_sets) >= 2:
        # Complete graph over cut events; weight(a,b) = lower bound on
        # Pr[D_a ∩ D_b] (capped by the smaller per-cut upper bound, since a joint
        # cannot exceed either marginal). All pairs are added -- including weight
        # 0 -- so the maximum-weight spanning tree spans every cut event, matching
        # Hunter's inequality exactly. Using a sub-forest would also be sound
        # (subtracting fewer terms) but looser.
        g = nx.Graph()
        g.add_nodes_from(range(len(cut_sets)))
        for a in range(len(cut_sets)):
            for b in range(a + 1, len(cut_sets)):
                rho = max(0.0, cut_joint_lower(cut_sets[a], cut_sets[b]))
                rho = min(rho, per_cut[a][1], per_cut[b][1])
                g.add_edge(a, b, weight=rho)
        mst = nx.maximum_spanning_tree(g)
        delta_hw = float(sum(d["weight"] for _, _, d in mst.edges(data=True)))
        tree_edges = [(u, v) for u, v in mst.edges()]

    # Subtract the Hunter--Worsley correction from the UNCAPPED sum, then clamp.
    # Capping the Boole sum to 1 *before* subtracting would be unsound (it could
    # push B_HW below the true union probability when raw_sum > 1).
    b_hw = max(0.0, min(1.0, raw_sum - delta_hw))
    return BoundResult(b_boole, b_hw, per_cut, delta_hw, tree_edges, regime)


# --------------------------------------------------------------------------
# Baseline composition rules used as comparators in the evaluation (Sec. VI).
# --------------------------------------------------------------------------
def multiplicative_certificate(cut_sets: list[frozenset[str]], eps: dict[str, float]) -> float:
    """The practitioner default (the ablation that breaks, Prop. 1).

    Treats component failures as INDEPENDENT: a cut's probability is the *product*
    of its members' marginals, and cuts are summed. This is NOT a sound upper
    bound under positive correlation; we compute it precisely to demonstrate when
    it is violated.
    """
    total = 0.0
    for cut in cut_sets:
        p = 1.0
        for i in cut:
            p *= eps.get(i, 1.0)
        total += p
    return total
