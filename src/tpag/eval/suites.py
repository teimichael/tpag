"""Suites for the TPAG replication package."""

from __future__ import annotations

from ..sim.workflows import redundancy_gadget, series_chain, t1_approval_workflow


def calibration_configs() -> list[tuple[str, object]]:
    """The manuscript's 19 base calibration configurations: redundancy gadgets across
    width/coupling, correlated series chains, and the T1 workflow at several couplings."""
    configs: list[tuple[str, object]] = []
    for k in (2, 3):
        for c in (0.0, 0.1, 0.2, 0.4, 0.6):
            configs.append(("gadget", redundancy_gadget(k=k, benign_rate=0.1, coupling=c)))
    for k in (3, 4):
        for c in (0.1, 0.3, 0.5):
            configs.append(("series", series_chain(k=k, benign_rate=0.08, coupling=c)))
    for c in (0.0, 0.2, 0.4):
        configs.append(("t1", t1_approval_workflow(coupling=c)))
    return configs


__all__ = ["calibration_configs"]
