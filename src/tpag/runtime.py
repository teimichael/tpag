"""Runtime for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass, field

from .causal import AssumptionCheck, CausalMonitor
from .certificate import Certificate, certify
from .containment import ContainmentController
from .contracts import Contract
from .estimation import EstimationModel
from .events import EventKind, Trace
from .graph import OrchestrationGraph
from .monitors import Verdict
from .structure import Structure


def _relevant_joints(structure: Structure, include_full_cuts: bool = False) -> set[frozenset[str]]:
    """Joint events whose co-violation the bound can use: pairs inside a common cut
    (Fréchet tightening) and pairs of singleton cuts (Hunter--Worsley series
    tightening). With ``include_full_cuts``, also the full member set of every cut
    of size >= 3, so the measured k-way co-violation tightens the cut directly
    (``bound._cut_upper``) instead of degrading to pairwise Fréchet."""
    joints: set[frozenset[str]] = set()
    cuts = structure.cut_sets()
    for cut in cuts:
        members = sorted(cut)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                joints.add(frozenset({members[i], members[j]}))
        if include_full_cuts and len(cut) >= 3:
            joints.add(cut)
    singletons = [next(iter(c)) for c in cuts if len(c) == 1]
    for i in range(len(singletons)):
        for j in range(i + 1, len(singletons)):
            joints.add(frozenset({singletons[i], singletons[j]}))
    return joints


@dataclass
class EpisodeResult:
    violated: dict[str, bool]
    discharged: dict[str, bool]
    system_failed: bool
    breaches: list[str] = field(default_factory=list)


class TPAGRuntime:
    def __init__(
        self,
        graph: OrchestrationGraph,
        structure: Structure,
        contracts: dict[str, Contract],
        assumption_checks: dict[str, AssumptionCheck] | None = None,
        containment: ContainmentController | None = None,
        regime: str = "series-parallel",
        register_cut_joints: bool = False,
    ) -> None:
        self.graph = graph
        self.structure = structure
        self.contracts = contracts
        self.assumption_checks = assumption_checks or {}
        self.containment = containment
        self.regime = regime

        self.estimation = EstimationModel()
        self.causal = CausalMonitor()
        # Default (pairs only) matches the committed evaluation runs exactly;
        # ``register_cut_joints=True`` additionally tracks the full k-way joint of
        # every cut of size >= 3 (strictly tighter, still sound; adds one estimated
        # quantity per such cut to the 1 - N*eta accounting).
        self.relevant_joints = _relevant_joints(structure, include_full_cuts=register_cut_joints)

        self.episodes = 0
        self.system_violations = 0
        self.monitor_log: list[dict] = []

    # -- ingestion -----------------------------------------------------------
    def ingest(self, trace: Trace, record_details: bool = False) -> EpisodeResult:
        violated: dict[str, bool] = {}
        discharged: dict[str, bool] = {}
        breaches: list[str] = []

        for node in self.graph.nodes:
            contract = self.contracts.get(node)
            node_events = [e for e in trace if e.lifeline == node]

            # (i) guarantee monitor -> Bernoulli outcome
            if contract is not None:
                gmon = contract.guarantee.monitor()
                gv = gmon.run(node_events)
                violated[node] = gv == Verdict.VIOLATED
            else:
                violated[node] = False

            # (ii) assumption discharge over the causal past
            act = next(
                (e for e in trace if e.lifeline == node and e.kind == EventKind.ACTIVATION),
                None,
            )
            check = self.assumption_checks.get(node)
            if act is not None and check is not None:
                dr = self.causal.discharged(trace, act, check)
                discharged[node] = dr.discharged
            else:
                discharged[node] = True  # no assumption / no activation -> vacuous

            if not discharged[node]:
                breaches.append(node)

            if record_details:
                self.monitor_log.append(
                    {
                        "episode": self.episodes,
                        "node": node,
                        "guarantee_violated": violated[node],
                        "assumption_discharged": discharged[node],
                    }
                )

        # (iii) estimation updates (only in-context activations count for eps_i)
        for node in self.graph.nodes:
            if discharged[node] and self.contracts.get(node) is not None:
                self.estimation.marginal(node).observe(violated[node])
        for members in self.relevant_joints:
            if all(discharged.get(m, False) for m in members):
                self.estimation.joint(members).observe(
                    all_in_context=True,
                    all_violated=all(violated.get(m, False) for m in members),
                )

        # system outcome via the failure structure
        vset = {n for n, v in violated.items() if v}
        system_failed = self.structure.failed(vset)
        self.episodes += 1
        self.system_violations += int(system_failed)

        # (iv) scoped containment on breach
        if self.containment is not None:
            for node in breaches:
                self.containment.contain(node)

        return EpisodeResult(violated, discharged, system_failed, breaches)

    def ingest_many(self, traces, record_details: bool = False) -> None:
        for t in traces:
            self.ingest(t, record_details=record_details)

    # -- outputs -------------------------------------------------------------
    @property
    def observed_violation_rate(self) -> float:
        return self.system_violations / self.episodes if self.episodes else float("nan")

    def certificate(self, eta: float = 0.01) -> Certificate:
        return certify(self.structure, self.estimation, eta=eta, regime=self.regime)
