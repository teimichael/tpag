"""Containment for the TPAG replication package."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import OrchestrationGraph


@dataclass
class ContainmentEvent:
    blamed_node: str
    scope: frozenset[str]
    kind: str  # "scoped" | "global"

    def contains(self, node: str) -> bool:
        return node in self.scope


class ContainmentController:
    """Scoped containment: blocks exactly the dominated sub-DAG of the blamed node."""

    kind = "scoped"

    def __init__(self, graph: OrchestrationGraph) -> None:
        self.graph = graph
        self._blocked: set[str] = set()
        self.events: list[ContainmentEvent] = []

    def contain(self, blamed_node: str) -> ContainmentEvent:
        scope = self.graph.dominated_subdag(blamed_node)
        ev = ContainmentEvent(blamed_node, frozenset(scope), self.kind)
        self._blocked |= scope
        self.events.append(ev)
        return ev

    def is_blocked(self, node: str) -> bool:
        return node in self._blocked

    # two-phase side-effect gate ------------------------------------------------
    def may_commit(self, node: str) -> bool:
        """A proposed side effect (SIDE_EFFECT_INTENT) on ``node`` may proceed to
        SIDE_EFFECT_COMMIT only if the node is not currently contained."""
        return not self.is_blocked(node)

    def reset(self) -> None:
        self._blocked.clear()
        self.events.clear()


class GlobalContainment(ContainmentController):
    """Baseline: any breach halts the entire workflow (whole-graph scope)."""

    kind = "global"

    def contain(self, blamed_node: str) -> ContainmentEvent:
        scope = set(self.graph.nodes)
        ev = ContainmentEvent(blamed_node, frozenset(scope), self.kind)
        self._blocked |= scope
        self.events.append(ev)
        return ev


@dataclass
class LocalizationScore:
    """Precision/recall of a containment scope against the ground-truth affected
    set (the nodes that genuinely depend on the breach)."""

    precision: float
    recall: float
    halted_fraction: float
    scope_size: int
    truth_size: int
    n_nodes: int = 0


def localization_score(
    scope: set[str], ground_truth_affected: set[str], n_nodes: int
) -> LocalizationScore:
    tp = len(scope & ground_truth_affected)
    precision = tp / len(scope) if scope else 1.0
    recall = tp / len(ground_truth_affected) if ground_truth_affected else 1.0
    return LocalizationScore(
        precision=precision,
        recall=recall,
        halted_fraction=len(scope) / n_nodes if n_nodes else 0.0,
        scope_size=len(scope),
        truth_size=len(ground_truth_affected),
        n_nodes=n_nodes,
    )
