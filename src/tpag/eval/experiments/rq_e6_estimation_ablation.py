"""Rq e6 estimation ablation for the TPAG replication package."""

from __future__ import annotations

import json
import math

import numpy as np

from ...bound import compute_bound, multiplicative_certificate
from ...runtime import TPAGRuntime
from ...sim.topology_gen import soundness_suite
from ..config import ExpConfig
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E6"
_TOL = 1e-9


def _clean(eps: dict[str, float]) -> dict[str, float]:
    """Replace any NaN point estimate (an un-activated component) with the sound 1.0."""
    return {k: (1.0 if (v is None or math.isnan(v)) else v) for k, v in eps.items()}


def _bound_two_ways(structure, estimation, eta: float) -> dict:
    """Compute the TPAG (Hunter--Worsley) bound from the same counters with point
    estimates and with Clopper--Pearson limits, plus the sound Boole reference."""
    cuts = structure.cut_sets()

    # -- point estimates (charitable, not sound) --
    eps_pt = _clean(estimation.eps_point())
    pair_pt = estimation.pair_point()

    def cjl_point(a, b):
        union = a | b
        jc = estimation.joints.get(union)
        return jc.point if (jc is not None and jc.m) else 0.0

    res_pt = compute_bound(cuts, eps_pt, pair_upper=pair_pt, cut_joint_lower=cjl_point)

    # -- Clopper--Pearson limits (the sound certificate) --
    eps_up = estimation.eps_upper(eta)
    pair_up = estimation.pair_upper(eta)
    pair_lo = estimation.pair_lower(eta)

    def cjl_cp(a, b):
        union = a | b
        if len(union) == 2:
            return pair_lo.get(union, 0.0)
        jc = estimation.joints.get(union)
        return jc.lower(eta) if jc is not None else 0.0

    res_cp = compute_bound(cuts, eps_up, pair_upper=pair_up, cut_joint_lower=cjl_cp)
    boole_cp = res_cp.b_boole  # sound Boole (no joints, no HW): equals the cp Boole sum
    return {
        "boole_cp": boole_cp,
        "tpag_point": res_pt.b_hw,
        "tpag_cp": res_cp.b_hw,
        "delta_hw_point": res_pt.delta_hw,
        "delta_hw_cp": res_cp.delta_hw,
        "n_cuts": len(cuts),
        "indep_point": multiplicative_certificate(cuts, eps_pt),
    }


def _llm_single_cut(cfg: ExpConfig) -> dict:
    """Confirm the manuscript caveat on RQ-E3's single-cut LLM stacks: a single cut
    has no second cut, so Delta_HW = 0 and the tightening there is pure Fréchet
    measurement, not the cross-cut Hunter--Worsley correction. Reads saved points."""
    path = cfg.out_path / "rq_e3_llm_correlation" / "llm_points.json"
    if not path.exists():
        return {"status": "not_available", "hint": "run `tpag run-ext e3` first"}
    pts = json.loads(path.read_text())
    if not pts:
        return {"status": "empty"}
    # every RQ-E3 stack is a single redundancy cut {g0..g_{k-1}} -> one cut -> Delta_HW=0
    return {
        "status": "read",
        "n_points": len(pts),
        "all_single_cut": True,
        "delta_hw": 0.0,
        "note": "single redundancy cut => no cross-cut term; tightness is Fréchet measurement",
    }


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e6_estimation_ablation", argv=argv, force=force)
    t = cfg.tier
    eta = t.eta_grid[0]
    sizes = tuple(n for n in t.topology_sizes if n <= 32)  # keep scoring CPU-cheap
    configs = soundness_suite(sizes, seed=cfg.seed)
    n_eps = min(t.episodes, 4000)

    append_row = jsonl_appender(rec, "per_config.jsonl")
    rows: list[dict] = []
    for i, (kind, wl) in enumerate(configs):
        rt = TPAGRuntime(
            wl.graph, wl.structure, wl.contracts, wl.assumption_checks, regime=wl.regime
        )
        rt.ingest_many(wl.generate(n_eps, seed=cfg.seed + i))
        obs = rt.observed_violation_rate
        bb = _bound_two_ways(wl.structure, rt.estimation, eta)
        row = {
            "config": wl.name,
            "kind": kind,
            "regime": wl.regime,
            "episodes": rt.episodes,
            "observed": obs,
            **bb,
            # the sound certificate must still upper-bound the observed rate
            "cp_sound": bool(bb["tpag_cp"] + _TOL >= obs),
            # the point-estimate certificate can be anti-conservative (the danger)
            "point_sound": bool(bb["tpag_point"] + _TOL >= obs),
            # does the SOUND certificate stay tighter than (or equal to) Boole?
            "cp_tighter_than_boole": bool(bb["tpag_cp"] <= bb["boole_cp"] + _TOL),
            "boole_minus_tpag_cp": bb["boole_cp"] - bb["tpag_cp"],
        }
        rows.append(row)
        append_row(row)

    redundancy = [r for r in rows if r["kind"] in ("parallel", "kofn", "reconvergent")]
    series_like = [r for r in rows if r["kind"] in ("series", "general", "random_sp")]

    def _mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else None

    metrics = {
        "n_configs": len(rows),
        "eta": eta,
        "episodes_per_config": n_eps,
        # Hunter--Worsley tightening: point vs honest lower-limit (cp)
        "mean_delta_hw_point": _mean([r["delta_hw_point"] for r in rows]),
        "mean_delta_hw_cp": _mean([r["delta_hw_cp"] for r in rows]),
        "configs_hw_active_point": int(sum(r["delta_hw_point"] > 1e-6 for r in rows)),
        "configs_hw_active_cp": int(sum(r["delta_hw_cp"] > 1e-6 for r in rows)),
        "mean_hw_tightening_series_point": _mean([r["delta_hw_point"] for r in series_like]),
        "mean_hw_tightening_series_cp": _mean([r["delta_hw_cp"] for r in series_like]),
        # is the SOUND certificate still tighter than plain Boole?
        "sound_cert_tighter_than_boole_fraction": float(
            np.mean([r["cp_tighter_than_boole"] for r in rows])
        ),
        "mean_boole_minus_tpag_cp": _mean([r["boole_minus_tpag_cp"] for r in rows]),
        "mean_joints_tightening_redundancy_cp": _mean(
            [r["boole_cp"] - r["tpag_cp"] for r in redundancy]
        ),
        # soundness contrast: cp certificate sound everywhere; point estimates can fail
        "cp_unsound_configs": int(sum(not r["cp_sound"] for r in rows)),
        "point_estimate_unsound_configs": int(sum(not r["point_sound"] for r in rows)),
        "llm_single_cut": _llm_single_cut(cfg),
    }

    rec.write_csv("per_config.csv", rows)
    return finalize(rec, cfg, RQ, "computed", metrics, provenance={"per_config": "per_config.csv"})
