"""Topology gen for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np

from ..causal import requires_upstream_label
from ..contracts import Contract
from ..graph import OrchestrationGraph
from ..predicates import always, guard_never_wrongly_approves
from ..structure import (
    Atom,
    ExplicitCutSets,
    Structure,
    kofn,
    parallel,
    series,
)
from .workflows import Workload

_TRIVIAL_ASSUMPTION = always(lambda e: True, name="true")

TOPOLOGY_FAMILIES = (
    "series",
    "parallel",
    "kofn",
    "reconvergent",
    "random_sp",
    "general",
)


# ---------------------------------------------------------------------------
# Correlation model + workload assembly.
# ---------------------------------------------------------------------------
def _blocked_sampler(
    benign: list[float], blocks: list[list[int]], coupling: float
) -> Callable[[np.random.Generator], list[bool]]:
    """Block shared-mode miss sampler. Within a block: with prob ``coupling`` all
    members miss together, else each misses independently at its benign rate."""

    def sampler(rng: np.random.Generator) -> list[bool]:
        out = [False] * len(benign)
        for block in blocks:
            if coupling > 0.0 and rng.random() < coupling:
                for i in block:
                    out[i] = True
            else:
                for i in block:
                    out[i] = bool(rng.random() < benign[i])
        return out

    return sampler


def _guard_contract(name: str, eps: float) -> Contract:
    return Contract(
        assumption=_TRIVIAL_ASSUMPTION,
        guarantee=guard_never_wrongly_approves(name=f"{name}_sound"),
        epsilon=eps,
        name=name,
    )


def _make_workload(
    name: str,
    structure: Structure,
    guard_names: list[str],
    benign: list[float],
    blocks: list[list[int]],
    coupling: float,
    regime: str,
) -> Workload:
    """Assemble a runnable :class:`Workload` from a failure structure.

    The orchestration graph is a flat ``planner -> guards -> actor`` fan; the
    *topology semantics live in the failure structure* (its cut sets), which is
    exactly what the composition bound consumes. Each guard emits an approve/block
    decision in its episode trace (via :func:`build_workflow_trace`), so the
    runtime's guarantee monitor recovers the per-guard violation, and
    ``structure.failed`` maps the violation set through the topology.
    """
    graph = OrchestrationGraph()
    for nd in ["planner", *guard_names, "actor"]:
        graph.add_component(nd)
    for g in guard_names:
        graph.add_edge("planner", g)
        graph.add_edge(g, "actor")
    contracts = {g: _guard_contract(g, b) for g, b in zip(guard_names, benign)}
    checks = {"actor": requires_upstream_label("approved")}
    sampler = _blocked_sampler(benign, blocks, coupling)
    return Workload(
        name=name,
        graph=graph,
        structure=structure,
        contracts=contracts,
        assumption_checks=checks,
        regime=regime,
        guard_names=guard_names,
        miss_sampler=sampler,
        benign_rates=list(benign),
        coupling=coupling,
    )


# ---------------------------------------------------------------------------
# Family builders (return runnable Workloads).
# ---------------------------------------------------------------------------
def series_topology(n: int, benign_rate: float = 0.06, coupling: float = 0.3) -> Workload:
    names = [f"s{i + 1}" for i in range(n)]
    structure = series(*names)
    blocks = [list(range(n))] if coupling > 0 else [[i] for i in range(n)]
    return _make_workload(
        f"series_n{n}_c{coupling}",
        structure,
        names,
        [benign_rate] * n,
        blocks,
        coupling,
        regime="series",
    )


def parallel_topology(n: int, benign_rate: float = 0.12, coupling: float = 0.3) -> Workload:
    names = [f"g{i + 1}" for i in range(n)]
    structure = parallel(*names)
    blocks = [list(range(n))] if coupling > 0 else [[i] for i in range(n)]
    return _make_workload(
        f"parallel_n{n}_c{coupling}",
        structure,
        names,
        [benign_rate] * n,
        blocks,
        coupling,
        regime="redundancy",
    )


def kofn_voting(k: int, n: int, benign_rate: float = 0.12, coupling: float = 0.3) -> Workload:
    names = [f"v{i + 1}" for i in range(n)]
    structure = kofn(k, *names)
    blocks = [list(range(n))] if coupling > 0 else [[i] for i in range(n)]
    return _make_workload(
        f"kofn_{k}of{n}_c{coupling}",
        structure,
        names,
        [benign_rate] * n,
        blocks,
        coupling,
        regime="series-parallel",
    )


def reconvergent_dag(
    width: int, depth: int, benign_rate: float = 0.1, coupling: float = 0.4
) -> Workload:
    """``depth`` redundant stages in series, each a ``width``-fold parallel block.

    Structure = ``series(parallel(stage1...), parallel(stage2...), ...)``. Within
    a stage the guards share a blind spot (one correlation block per stage);
    stages are independent. ``N = width * depth`` guards."""
    names: list[str] = []
    blocks: list[list[int]] = []
    stages: list[Structure] = []
    idx = 0
    for d in range(depth):
        stage = [f"r{d + 1}_{w + 1}" for w in range(width)]
        blocks.append(list(range(idx, idx + width)))
        idx += width
        names += stage
        stages.append(parallel(*stage))
    structure = series(*stages)
    return _make_workload(
        f"reconvergent_w{width}d{depth}_c{coupling}",
        structure,
        names,
        [benign_rate] * len(names),
        blocks,
        coupling,
        regime="series-parallel",
    )


def _random_sp_structure(names: list[str], rng: np.random.Generator) -> Structure:
    if len(names) == 1:
        return Atom(names[0])
    cut = int(rng.integers(1, len(names)))
    left = _random_sp_structure(names[:cut], rng)
    right = _random_sp_structure(names[cut:], rng)
    op = series if rng.random() < 0.5 else parallel
    return op(left, right)


def random_series_parallel(
    n: int, seed: int, benign_rate: float = 0.1, coupling: float = 0.25
) -> Workload:
    rng = np.random.default_rng(seed)
    names = [f"a{i + 1}" for i in range(n)]
    structure = _random_sp_structure(names, rng)
    blocks = [list(range(n))] if coupling > 0 else [[i] for i in range(n)]
    return _make_workload(
        f"random_sp_n{n}_s{seed}",
        structure,
        names,
        [benign_rate] * n,
        blocks,
        coupling,
        regime="series-parallel",
    )


def random_general_dag(
    n: int,
    seed: int,
    benign_rate: float = 0.1,
    coupling: float = 0.2,
    n_cuts: int | None = None,
    max_cut_size: int = 3,
) -> Workload:
    """A random *monotone* (coherent) structure declared by explicit minimal cut
    sets -- the general, non-series-parallel regime. ``n_cuts`` defaults to ``2n``."""
    rng = np.random.default_rng(seed)
    names = [f"a{i + 1}" for i in range(n)]
    target = n_cuts if n_cuts is not None else 2 * n
    cuts: set[frozenset[str]] = set()
    guard = 0
    while len(cuts) < target and guard < target * 50:
        guard += 1
        size = int(rng.integers(2, max_cut_size + 1))
        members = rng.choice(names, size=min(size, n), replace=False)
        cuts.add(frozenset(members.tolist()))
    structure = ExplicitCutSets(list(cuts))
    # correlation blocks = the cut sets themselves (intra-cut shared blind spots)
    name_idx = {nm: i for i, nm in enumerate(names)}
    blocks = (
        [[name_idx[m] for m in cut] for cut in cuts] if coupling > 0 else [[i] for i in range(n)]
    )
    return _make_workload(
        f"general_n{n}_s{seed}",
        structure,
        names,
        [benign_rate] * n,
        blocks,
        coupling,
        regime="general",
    )


# ---------------------------------------------------------------------------
# Suites (enumerate the configs an RQ driver sweeps).
# ---------------------------------------------------------------------------
def soundness_suite(sizes: tuple[int, ...], seed: int = 0) -> list[tuple[str, Workload]]:
    """A topology-and-scale sweep for RQ-E2 soundness/tightness. Mixes correlated
    and independent variants across the tractable families at each ``N``."""
    out: list[tuple[str, Workload]] = []
    for n in sizes:
        for c in (0.0, 0.3):
            out.append(("series", series_topology(n, coupling=c)))
            out.append(("parallel", parallel_topology(n, coupling=c)))
        out.append(("random_sp", random_series_parallel(n, seed=seed + n, coupling=0.25)))
        out.append(("general", random_general_dag(n, seed=seed + n, coupling=0.2)))
    # k-of-n voting at a few small, fixed shapes (combinatorial cuts kept tractable)
    for k, n in ((2, 3), (3, 5), (3, 7), (4, 7)):
        for c in (0.0, 0.3):
            out.append(("kofn", kofn_voting(k, n, coupling=c)))
    # reconvergent diamond stacks
    for w, d in ((2, 2), (3, 3), (2, 5), (4, 4)):
        out.append(("reconvergent", reconvergent_dag(w, d, coupling=0.4)))
    return out


def estimated_cut_count(family: str, n: int) -> float:
    """Closed-form estimate of the minimal-cut-set count, used to *skip* configs
    that would blow up enumeration (the fail-safe for the scaling curve)."""
    if family == "series":
        return float(n)
    if family == "parallel":
        return 1.0
    if family == "reconvergent":
        return float(n // 2)
    if family == "kofn":
        k = max(1, n // 2)
        return float(comb(n, max(1, n - k + 1)))
    if family == "nested_parallel":
        return float(2 ** (n // 2))
    if family == "general":
        return float(2 * n)
    return float(n)


def scaling_structure(family: str, n: int) -> Structure:
    """Structure-only builder for the cut-set/bound timing curve (RQ-E2)."""
    names = [f"x{i + 1}" for i in range(n)]
    if family == "series":
        return series(*names)
    if family == "parallel":
        return parallel(*names)
    if family == "kofn":
        return kofn(max(1, n // 2), *names)
    if family == "reconvergent":
        stages = [parallel(names[i], names[i + 1]) for i in range(0, n - 1, 2)]
        return series(*stages) if stages else Atom(names[0])
    if family == "nested_parallel":
        pairs = [series(names[i], names[i + 1]) for i in range(0, n - 1, 2)]
        return parallel(*pairs) if pairs else Atom(names[0])
    if family == "general":
        return random_general_dag(n, seed=n).structure
    raise ValueError(f"unknown scaling family {family!r}")


SCALING_FAMILIES = ("series", "parallel", "reconvergent", "kofn", "nested_parallel", "general")


def scaling_suite(sizes: tuple[int, ...], max_cuts: int) -> list[dict]:
    """Enumerate ``(family, n)`` timing points, marking which exceed ``max_cuts``
    (so the harness skips enumeration rather than OOM-ing)."""
    points: list[dict] = []
    for family in SCALING_FAMILIES:
        for n in sizes:
            est = estimated_cut_count(family, n)
            points.append(
                {
                    "family": family,
                    "n": n,
                    "estimated_cuts": est,
                    "skip": est > max_cuts,
                }
            )
    return points


__all__ = [
    "TOPOLOGY_FAMILIES",
    "SCALING_FAMILIES",
    "series_topology",
    "parallel_topology",
    "kofn_voting",
    "reconvergent_dag",
    "random_series_parallel",
    "random_general_dag",
    "soundness_suite",
    "scaling_suite",
    "scaling_structure",
    "estimated_cut_count",
]
