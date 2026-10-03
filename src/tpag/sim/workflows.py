"""Workflows for the TPAG replication package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..causal import AssumptionCheck, requires_upstream_label
from ..contracts import Contract
from ..events import Trace
from ..graph import OrchestrationGraph
from ..predicates import always, guard_never_wrongly_approves
from ..seeding import SeedManager
from ..structure import Structure, check_adequacy, parallel, series
from .components import build_workflow_trace, sample_correlated_misses

_TRIVIAL_ASSUMPTION = always(lambda e: True, name="true")


def _guard_contract(name: str, eps: float) -> Contract:
    return Contract(
        assumption=_TRIVIAL_ASSUMPTION,
        guarantee=guard_never_wrongly_approves(name=f"{name}_sound"),
        epsilon=eps,
        name=name,
    )


@dataclass
class Workload:
    name: str
    graph: OrchestrationGraph
    structure: Structure
    contracts: dict[str, Contract]
    assumption_checks: dict[str, AssumptionCheck]
    regime: str
    guard_names: list[str]
    miss_sampler: Callable[[np.random.Generator], list[bool]]
    benign_rates: list[float] = field(default_factory=list)
    coupling: float = 0.0
    pii: bool = False

    @property
    def declared_components(self) -> set[str]:
        return self.structure.atoms()

    def adequacy(self):
        return check_adequacy(self.structure, self.declared_components)

    def generate(self, n: int, seed: int) -> list[Trace]:
        rng = SeedManager(seed).rng(self.name)
        traces = []
        for _ in range(n):
            misses = self.miss_sampler(rng)
            traces.append(
                build_workflow_trace(self.guard_names, misses, pii=self.pii, redaction_ok=True)
            )
        return traces

    def true_marginals(self) -> dict[str, float]:
        """The analytic marginal miss rate per guard under the shared-mode model
        (for reference/validation, not used by the certificate)."""
        c = self.coupling
        return {name: c + (1 - c) * q for name, q in zip(self.guard_names, self.benign_rates)}


def redundancy_gadget(
    k: int = 2, benign_rate: float = 0.1, coupling: float = 0.3, prefix: str = "g"
) -> Workload:
    guard_names = [f"{prefix}{i + 1}" for i in range(k)]
    benign = [benign_rate] * k
    graph = OrchestrationGraph()
    for n in ["planner", *guard_names, "actor"]:
        graph.add_component(n)
    for g in guard_names:
        graph.add_edge("planner", g)
        graph.add_edge(g, "actor")
    structure = parallel(*guard_names)
    contracts = {g: _guard_contract(g, benign_rate) for g in guard_names}
    checks = {"actor": requires_upstream_label("approved")}

    def sampler(rng: np.random.Generator) -> list[bool]:
        return sample_correlated_misses(benign, coupling, rng)

    return Workload(
        name=f"gadget_k{k}_c{coupling}",
        graph=graph,
        structure=structure,
        contracts=contracts,
        assumption_checks=checks,
        regime="redundancy",
        guard_names=guard_names,
        miss_sampler=sampler,
        benign_rates=benign,
        coupling=coupling,
    )


def series_chain(
    k: int = 3, benign_rate: float = 0.08, coupling: float = 0.3, prefix: str = "s"
) -> Workload:
    """A pure delegation chain of ``k`` guards (all singleton cuts) with positively
    correlated failures -- the setting where Hunter--Worsley strictly tightens Boole
    (Cor. 1). ``not Phi`` holds if ANY link fails."""
    guard_names = [f"{prefix}{i + 1}" for i in range(k)]
    benign = [benign_rate] * k
    graph = OrchestrationGraph()
    chain = ["planner", *guard_names, "actor"]
    for n in chain:
        graph.add_component(n)
    for a, b in zip(chain, chain[1:]):
        graph.add_edge(a, b)
    structure = series(*guard_names)
    contracts = {g: _guard_contract(g, benign_rate) for g in guard_names}
    checks = {"actor": requires_upstream_label("approved")}

    def sampler(rng: np.random.Generator) -> list[bool]:
        return sample_correlated_misses(benign, coupling, rng)

    return Workload(
        name=f"series_k{k}_c{coupling}",
        graph=graph,
        structure=structure,
        contracts=contracts,
        assumption_checks=checks,
        regime="series",
        guard_names=guard_names,
        miss_sampler=sampler,
        benign_rates=benign,
        coupling=coupling,
    )


def t1_approval_workflow(
    policy_benign: float = 0.12,
    pii_benign: float = 0.05,
    coupling: float = 0.3,
) -> Workload:
    guard_names = ["policy1", "policy2", "pii"]
    graph = OrchestrationGraph()
    for n in ["planner", "specialist", *guard_names, "actor"]:
        graph.add_component(n)
    graph.add_edge("planner", "specialist")
    for g in guard_names:
        graph.add_edge("specialist", g)
        graph.add_edge(g, "actor")
    # not Phi = (policy1 AND policy2 miss) OR (pii misses)
    structure = series(parallel("policy1", "policy2"), "pii")
    contracts = {
        "policy1": _guard_contract("policy1", policy_benign),
        "policy2": _guard_contract("policy2", policy_benign),
        "pii": _guard_contract("pii", pii_benign),
    }
    checks = {"actor": requires_upstream_label("approved")}

    def sampler(rng: np.random.Generator) -> list[bool]:
        pol = sample_correlated_misses([policy_benign, policy_benign], coupling, rng)
        pii_miss = bool(rng.random() < pii_benign)
        return [*pol, pii_miss]

    return Workload(
        name=f"t1_c{coupling}",
        graph=graph,
        structure=structure,
        contracts=contracts,
        assumption_checks=checks,
        regime="series-parallel",
        guard_names=guard_names,
        miss_sampler=sampler,
        benign_rates=[policy_benign, policy_benign, pii_benign],
        coupling=coupling,
        pii=True,
    )
