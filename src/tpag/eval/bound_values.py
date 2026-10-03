"""Bound values for the TPAG replication package."""

from __future__ import annotations
import statistics as st
from typing import Any

THRESHOLDS = (0.05, 0.1, 0.25, 0.5)


def _bounds(points: list[dict]) -> list[float]:
    return [float(p["tpag_bound"]) for p in points if p.get("tpag_bound") is not None]


def _group_stats(points: list[dict]) -> dict[str, Any]:
    """Bound-value stats for one (sub)group of points."""
    b = _bounds(points)
    if not b:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": len(b),
        "min_bound": min(b),
        "median_bound": st.median(b),
        "mean_bound": st.fmean(b),
    }
    for thr in THRESHOLDS:
        out[f"frac_le_{thr:g}"] = sum((x <= thr + 1e-12 for x in b)) / len(b)
    return out


def _non_vacuous_exemplar(points: list[dict]) -> dict | None:
    """A concrete, citable non-vacuous certification: the cross-family point with the
    smallest certified bound, preferring hardened guards and -- among ties -- the
    *highest* injection fraction (a small bound that survives injection is the
    strongest evidence the framework certifies, not only refuses). Falls back to any
    point if no cross-family point exists."""
    cand = [p for p in points if p.get("tpag_bound") is not None]
    if not cand:
        return None
    cross = [p for p in cand if p.get("diversity") == "cross_family"] or cand

    def key(p: dict) -> tuple:
        hardened = 0 if p.get("hardening") in ("strict", "hardened") else 1
        return (
            round(float(p["tpag_bound"]), 6),
            hardened,
            -float(p.get("injection_fraction", 0.0)),
        )

    best = min(cross, key=key)
    return {
        "models": best.get("models"),
        "diversity": best.get("diversity"),
        "hardening": best.get("hardening"),
        "k": best.get("k"),
        "domain": best.get("domain"),
        "strategy": best.get("strategy"),
        "injection_fraction": best.get("injection_fraction"),
        "tpag_bound": best.get("tpag_bound"),
        "confidence": best.get("confidence"),
        "observed_comiss": best.get("observed_comiss"),
        "n_samples": best.get("n_samples"),
    }


def summarize_bound_values(points: list[dict]) -> dict[str, Any]:
    """Compute the RQ-E3(d) certified-bound-value summary over LLM operating points.

    Returns the floor, central tendency, percentiles, the fraction of points certified
    at-or-below each threshold in :data:`THRESHOLDS`, per-diversity / per-hardening /
    per-(diversity x hardening) breakdowns, a ``cross_family_hardened`` rollup, and a
    concrete ``non_vacuous_exemplar``. Empty input yields ``{"n_points": 0}``."""
    b = _bounds(points)
    if not b:
        return {"n_points": 0}
    qs = st.quantiles(b, n=100) if len(b) >= 2 else [b[0]] * 99
    summary: dict[str, Any] = {
        "n_points": len(b),
        "floor": min(b),
        "max": max(b),
        "mean": st.fmean(b),
        "median": st.median(b),
        "percentiles": {f"p{p}": qs[p - 1] for p in (10, 25, 50, 75, 90)},
        "frac_le": {f"{thr:g}": sum((x <= thr + 1e-12 for x in b)) / len(b) for thr in THRESHOLDS},
        "by_diversity": {
            div: _group_stats([p for p in points if p.get("diversity") == div])
            for div in sorted({str(p.get("diversity")) for p in points})
        },
        "by_hardening": {
            h: _group_stats([p for p in points if p.get("hardening") == h])
            for h in sorted({str(p.get("hardening")) for p in points})
        },
        "by_diversity_hardening": {
            f"{div}|{h}": _group_stats(
                [p for p in points if p.get("diversity") == div and p.get("hardening") == h]
            )
            for div in sorted({str(p.get("diversity")) for p in points})
            for h in sorted({str(p.get("hardening")) for p in points})
        },
        "cross_family_hardened": _group_stats(
            [
                p
                for p in points
                if p.get("diversity") == "cross_family"
                and p.get("hardening") in ("strict", "hardened")
            ]
        ),
        "non_vacuous_exemplar": _non_vacuous_exemplar(points),
    }
    return summary


__all__ = ["summarize_bound_values"]
