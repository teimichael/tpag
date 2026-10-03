"""Graph for the TPAG replication package."""

from __future__ import annotations

import networkx as nx


class OrchestrationGraph:
    """A directed acyclic orchestration graph over component ids."""

    VIRTUAL_SOURCE = "__source__"

    def __init__(self) -> None:
        self._g = nx.DiGraph()

    # -- construction --------------------------------------------------------
    def add_component(self, node: str) -> None:
        if node == self.VIRTUAL_SOURCE:
            raise ValueError("reserved node id")
        self._g.add_node(node)

    def add_edge(self, src: str, dst: str) -> None:
        """Add dependency ``src -> dst`` (src is a predecessor/source of dst)."""
        self._g.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(self._g):
            self._g.remove_edge(src, dst)
            raise ValueError(f"edge {src}->{dst} would create a cycle; Gamma must be acyclic")

    # -- queries -------------------------------------------------------------
    @property
    def nodes(self) -> list[str]:
        return list(self._g.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self._g.edges)

    def pre(self, node: str) -> list[str]:
        """Immediate predecessors of ``node`` (the ``pre(i)`` of Def. 7)."""
        return list(self._g.predecessors(node))

    def successors(self, node: str) -> list[str]:
        return list(self._g.successors(node))

    def sources(self) -> list[str]:
        return [n for n in self._g.nodes if self._g.in_degree(n) == 0]

    def sinks(self) -> list[str]:
        return [n for n in self._g.nodes if self._g.out_degree(n) == 0]

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._g))

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    # -- dominators / containment -------------------------------------------
    def _with_virtual_source(self) -> nx.DiGraph:
        g = self._g.copy()
        g.add_node(self.VIRTUAL_SOURCE)
        for s in self.sources():
            g.add_edge(self.VIRTUAL_SOURCE, s)
        return g

    def dominated_subdag(self, node: str) -> set[str]:
        """Nodes dominated by ``node`` (every entry->n path passes through node).

        Returned set includes ``node`` itself. This is the scope a containment
        action blocks/rolls back when ``node``'s assumption is breached: exactly
        the part of the workflow that causally depends on the blamed node through
        every route, leaving incomparable branches running (RQ4).
        """
        if node not in self._g:
            raise KeyError(node)
        g = self._with_virtual_source()
        idom = nx.immediate_dominators(g, self.VIRTUAL_SOURCE)
        # Build dominator tree (child -> idom[child]) and collect subtree at node.
        dominated = set()
        for n in g.nodes:
            if n == self.VIRTUAL_SOURCE:
                continue
            cur = n
            seen = set()
            while cur != self.VIRTUAL_SOURCE and cur not in seen:
                seen.add(cur)
                if cur == node:
                    dominated.add(n)
                    break
                nxt = idom.get(cur)
                if nxt is None or nxt == cur:
                    break
                cur = nxt
        dominated.add(node)
        return dominated

    def descendants(self, node: str) -> set[str]:
        return set(nx.descendants(self._g, node))

    def to_networkx(self) -> nx.DiGraph:
        return self._g.copy()
