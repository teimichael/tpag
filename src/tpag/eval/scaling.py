"""Scaling for the TPAG replication package."""

from __future__ import annotations

import tracemalloc
from time import perf_counter

from ..bound import compute_bound
from ..sim.topology_gen import (
    SCALING_FAMILIES,
    estimated_cut_count,
    scaling_structure,
)

__all__ = ["SCALING_FAMILIES", "run_scaling_point", "run_scaling_curve"]


def run_scaling_point(
    family: str,
    n: int,
    *,
    rate: float = 0.05,
    max_cuts: int = 2_000_000,
    timeout_s: float = 30.0,
) -> dict:
    """Time cut-set enumeration + bound computation for one ``(family, N)``.

    Returns a record with ``status`` in {``computed``, ``skipped_cap``}. The
    ``skipped_cap`` status is emitted *without* enumerating, charting the boundary
    of the tractable regime.
    """
    est = estimated_cut_count(family, n)
    base = {
        "family": family,
        "n": n,
        "estimated_cuts": est,
        "max_cuts": max_cuts,
    }
    if est > max_cuts:
        return {
            **base,
            "status": "skipped_cap",
            "n_cuts": None,
            "cut_time_s": None,
            "bound_time_s": None,
            "peak_mib": None,
            "over_timeout": None,
        }

    structure = scaling_structure(family, n)
    tracemalloc.start()
    t0 = perf_counter()
    cuts = structure.cut_sets()
    cut_time = perf_counter() - t0

    eps = {a: rate for a in structure.atoms()}
    t1 = perf_counter()
    compute_bound(cuts, eps)
    bound_time = perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        **base,
        "status": "computed",
        "n_cuts": len(cuts),
        "cut_time_s": cut_time,
        "bound_time_s": bound_time,
        "peak_mib": peak / (2**20),
        "over_timeout": bool((cut_time + bound_time) > timeout_s),
    }


def run_scaling_curve(
    sizes: tuple[int, ...],
    *,
    families: tuple[str, ...] = SCALING_FAMILIES,
    max_cuts: int = 2_000_000,
    timeout_s: float = 30.0,
    rate: float = 0.05,
) -> list[dict]:
    """Sweep every ``(family, N)`` and return the timing records. Once a family is
    skipped at some ``N`` (cap exceeded), larger ``N`` for that family is also
    skipped (monotone), avoiding wasted estimation."""
    rows: list[dict] = []
    for family in families:
        capped = False
        for n in sizes:
            if capped:
                rows.append(
                    {
                        "family": family,
                        "n": n,
                        "estimated_cuts": estimated_cut_count(family, n),
                        "max_cuts": max_cuts,
                        "status": "skipped_cap",
                        "n_cuts": None,
                        "cut_time_s": None,
                        "bound_time_s": None,
                        "peak_mib": None,
                        "over_timeout": None,
                    }
                )
                continue
            rec = run_scaling_point(family, n, rate=rate, max_cuts=max_cuts, timeout_s=timeout_s)
            rows.append(rec)
            if rec["status"] == "skipped_cap" or rec.get("over_timeout"):
                capped = True
    return rows
