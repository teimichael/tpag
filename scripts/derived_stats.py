"""Derive the reported manuscript statistics from released evidence without model calls."""

from __future__ import annotations
import gzip
import numpy as np
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from tpag.bound import compute_bound
from tpag.estimation import clopper_pearson_lower, clopper_pearson_upper

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TIGHT = ROOT / "results_tighten"
OUT = ROOT / "outputs" / "derived" / "derived_stats.json"
SPLITS_OUT = ROOT / "outputs" / "derived" / "llm_heldout_splits.json"
EPS = 1e-09
_TOL = EPS
ETA_CERT = 0.001
ETA_E4 = 0.01
CI95 = 0.025


def load_json(p: Path):
    return json.loads(p.read_text())


def load_jsonl(p: Path):
    with p.open() as stream:
        return [json.loads(line) for line in stream]


def load_csv(p: Path) -> list[dict]:
    with p.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def mean(xs):
    xs = list(xs)
    return float(st.mean(xs)) if xs else None


def median(xs):
    xs = list(xs)
    return float(st.median(xs)) if xs else None


def ci_robust(observed: float, episodes: float, reference: float) -> bool:
    """Lower end of the two-sided 95% CP interval of ``observed`` exceeds ``reference``."""
    k = int(round(observed * episodes))
    return clopper_pearson_lower(k, int(episodes), CI95) > reference + EPS


def e1() -> dict:
    rows = load_csv(RES / "rq_e1_calibration/per_config.csv")
    sat = [r for r in rows if r["b_hw"] >= 1 - EPS]
    nonsat = [r for r in rows if r["b_hw"] < 1 - EPS]
    single = [r for r in rows if r["regime"] == "redundancy"]
    hw = [r for r in rows if r["b_boole"] - r["b_hw"] > EPS]
    hold = load_json(RES / "rq_e1_calibration/holdout.json")
    hold_nv = [h for h in hold if h["bound_train"] < 1 - EPS]
    eta = load_json(RES / "rq_e1_calibration/eta_sweep.json")
    return {
        "n_configs": len(rows),
        "episodes": rows[0]["episodes"],
        "in_sample_covered": sum((r["b_hw"] + EPS >= r["observed"] for r in rows)),
        "bound_ge_bootstrap_hi": sum((r["b_hw"] + EPS >= r["obs_ci_hi"] for r in rows)),
        "mean_slack": mean((r["slack"] for r in rows)),
        "saturated": len(sat),
        "nonsaturated_mean_slack": mean((r["slack"] for r in nonsat)),
        "nonsaturated_median_slack": median((r["slack"] for r in nonsat)),
        "single_cut_n": len(single),
        "single_cut_mean_slack": mean((r["slack"] for r in single)),
        "single_cut_max_slack": max((r["slack"] for r in single)),
        "confidence_min": min((r["confidence"] for r in rows)),
        "confidence_max": max((r["confidence"] for r in rows)),
        "hw_active": len(hw),
        "hw_active_mean_gain": mean((r["b_boole"] - r["b_hw"] for r in hw)),
        "hw_mean_gain_all": mean((r["b_boole"] - r["b_hw"] for r in rows)),
        "holdout_n": len(hold),
        "holdout_covered": sum((h["covered_out_of_sample"] for h in hold)),
        "holdout_nonvacuous": len(hold_nv),
        "holdout_min_margin_nonvacuous": min(
            (h["bound_train"] - h["observed_test"] for h in hold_nv)
        ),
        "holdout_episodes": [hold[0]["n_train"], hold[0]["n_test"]],
        "eta_sweep_mean_slack": {str(e["eta"]): e["mean_slack"] for e in eta},
        "eta_sweep_coverage": {str(e["eta"]): e["coverage"] for e in eta},
    }


def n_est_pairs(n: int) -> int:
    """Marginals plus all registered pairs for a series or parallel structure of n guards."""
    return n + n * (n - 1) // 2


def e2() -> dict:
    rows = load_csv(RES / "rq_e2_topology/per_config.csv")
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    per_family = {}
    for f, rs in fams.items():
        nv = [r for r in rs if r["b_hw"] < 1 - EPS]
        per_family[f] = {
            "configs": len(rs),
            "n_min": min((r["n_components"] for r in rs)),
            "n_max": max((r["n_components"] for r in rs)),
            "nonvacuous": len(nv),
            "mean_slack_all": mean((r["slack"] for r in rs)),
            "mean_slack_nonvacuous": mean((r["slack"] for r in nv)),
            "indep_unsound": sum((r["observed"] > r["independence"] + EPS for r in rs)),
            "indep_unsound_ci95": sum(
                (ci_robust(r["observed"], r["episodes"], r["independence"]) for r in rs)
            ),
            "in_sample_covered": sum((r["b_hw"] + EPS >= r["observed"] for r in rs)),
        }
    par = sorted((r for r in rows if r["family"] == "parallel"), key=lambda r: r["n_components"])
    par_c3 = [r for r in par if r["config"].endswith("c0.3")]
    par_c0 = [r for r in par if r["config"].endswith("c0.0")]
    by_cfg = {r["config"]: r for r in rows}
    scal = load_csv(RES / "rq_e2_topology/scaling.csv")
    comp = [r for r in scal if r["status"] == "computed"]
    at128 = [r for r in comp if r["n"] == 128]

    def total(r):
        return r["cut_time_s"] + r["bound_time_s"]

    def pick(fam, n):
        r = next((r for r in comp if r["family"] == fam and r["n"] == n))
        return {"n_cuts": r["n_cuts"], "seconds": total(r), "peak_mib": r["peak_mib"]}

    return {
        "n_configs": len(rows),
        "per_family": per_family,
        "indep_unsound_total": sum((v["indep_unsound"] for v in per_family.values())),
        "indep_unsound_ci95_total": sum((v["indep_unsound_ci95"] for v in per_family.values())),
        "nonvacuous_total": sum((v["nonvacuous"] for v in per_family.values())),
        "parallel_c03": [
            {
                "n": r["n_components"],
                "observed": r["observed"],
                "independence": r["independence"],
                "ratio": r["observed"] / r["independence"],
                "bound": r["b_hw"],
            }
            for r in par_c3
        ],
        "parallel_c0_indep_unsound": sum((r["observed"] > r["independence"] + EPS for r in par_c0)),
        "parallel_c0_bounds": [r["b_hw"] for r in par_c0],
        "hw_examples": {
            c: {
                "boole": by_cfg[c]["b_boole"],
                "hw": by_cfg[c]["b_hw"],
                "observed": by_cfg[c]["observed"],
            }
            for c in ("series_n4_c0.3", "series_n8_c0.3")
            if c in by_cfg
        },
        "confidence_by_n_series_parallel": {
            n: max(0.0, 1 - n_est_pairs(n) * ETA_CERT) for n in (4, 8, 16, 32, 64)
        },
        "scaling_points": len(scal),
        "scaling_computed": len(comp),
        "scaling_skipped_cap": sum((r["status"] == "skipped_cap" for r in scal)),
        "scaling_max_seconds_n128": max((total(r) for r in at128)),
        "scaling_max_cuts_n128": max((r["n_cuts"] for r in at128)),
        "scaling_families_n128": sorted({r["family"] for r in at128}),
        "kofn_n16": pick("kofn", 16),
        "nested_parallel_n32": pick("nested_parallel", 32),
        "max_n_by_family": {
            f: max((r["n"] for r in comp if r["family"] == f))
            for f in sorted({r["family"] for r in comp})
        },
    }


def llm_suite(points: list[dict]) -> dict:
    viol = [p for p in points if p["independence_violated"]]
    worst = max(viol, key=lambda p: p["underreport_ratio"])
    cp_robust = sum(
        (ci_robust(p["observed_comiss"], p["n_samples"], p["multiplicative"]) for p in points)
    )

    def axis(key, fn=None):
        groups = defaultdict(list)
        for p in points:
            groups[fn(p) if fn else p[key]].append(p)
        return {
            str(g): {
                "n": len(ps),
                "comiss": mean((p["observed_comiss"] for p in ps)),
                "product": mean((p["multiplicative"] for p in ps)),
                "viol": mean((p["independence_violated"] for p in ps)),
                "viol_ci": mean((p["independence_violated_ci"] for p in ps)),
                "median_bound": median((p["tpag_bound"] for p in ps)),
                "frac_bound_le_0.10": mean((p["tpag_bound"] <= 0.1 + EPS for p in ps)),
            }
            for g, ps in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }

    bounds = [p["tpag_bound"] for p in points]
    return {
        "n_points": len(points),
        "n_groups": len({p["group"] for p in points}),
        "n_samples": sorted({p["n_samples"] for p in points}),
        "n_samples_counts": {
            str(n): sum((p["n_samples"] == n for p in points))
            for n in sorted({p["n_samples"] for p in points})
        },
        "injection_fractions": sorted({p["injection_fraction"] for p in points}),
        "indep_violated": len(viol),
        "indep_violated_ci_bootstrap": sum((p["independence_violated_ci"] for p in points)),
        "indep_violated_ci_cp95": cp_robust,
        "in_sample_covered": sum((p["tpag_holds"] for p in points)),
        "max_underreport": {
            "ratio": worst["underreport_ratio"],
            "group": worst["group"],
            "injection_fraction": worst["injection_fraction"],
            "product": worst["multiplicative"],
            "observed": worst["observed_comiss"],
        },
        "confidence": sorted({p["confidence"] for p in points}),
        "bound_floor": min(bounds),
        "bound_median": median(bounds),
        "frac_bound_le_0.05": mean((b <= 0.05 + EPS for b in bounds)),
        "frac_bound_le_0.10": mean((b <= 0.1 + EPS for b in bounds)),
        "by_diversity": axis("diversity"),
        "by_k": axis("k"),
        "by_tier": axis("size_tier"),
        "by_hardening": axis("hardening"),
        "by_domain": axis("domain"),
        "by_strategy": axis("strategy"),
        "by_diversity_hardening": axis(None, lambda p: f"{p['diversity']}|{p['hardening']}"),
    }


def find(points, group, frac):
    return next(
        (p for p in points if p["group"] == group and abs(p["injection_fraction"] - frac) < EPS)
    )


def e3() -> dict:
    main = load_json(RES / "rq_e3_llm_correlation/llm_points.json")
    tight = load_json(TIGHT / "rq_e3_llm_correlation/llm_points.json")
    out = {"main": llm_suite(main), "tighten": llm_suite(tight)}

    def brief(p):
        return {
            k: p[k]
            for k in (
                "observed_comiss",
                "comiss_ci",
                "multiplicative",
                "tpag_bound",
                "n_samples",
                "confidence",
                "independence_violated_ci",
            )
        }

    g15 = "same[deepseek-r1:1.5b]x2|financial|strict|shared"
    out["examples"] = {
        "r1_1.5b_pair_strict_af0": brief(find(main, g15, 0.0)),
        "r1_1.5b_pair_strict_af1": brief(find(main, g15, 1.0)),
        "cross_r1_8b_granite_8b_strict_af1": brief(
            find(main, "cross[deepseek-r1:8b+granite4.1:8b]|financial|strict|shared", 1.0)
        ),
        "cross_r1_8b_granite_8b_permissive_af0": brief(
            find(main, "cross[deepseek-r1:8b+granite4.1:8b]|financial|permissive|shared", 0.0)
        ),
        "cross_r1_8b_granite_8b_permissive_af1": brief(
            find(main, "cross[deepseek-r1:8b+granite4.1:8b]|financial|permissive|shared", 1.0)
        ),
    }
    floor_t = min(tight, key=lambda p: p["tpag_bound"])
    out["tighten_floor_example"] = {
        "group": floor_t["group"],
        "injection_fraction": floor_t["injection_fraction"],
        **brief(floor_t),
    }
    idx = {(p["group"], p["injection_fraction"]): p for p in main}
    pairs = [
        (idx[t["group"], t["injection_fraction"]], t)
        for t in tight
        if (t["group"], t["injection_fraction"]) in idx
    ]
    borderline = [
        (m, t)
        for m, t in pairs
        if m["independence_violated"] and (not m["independence_violated_ci"])
    ]
    changed = [(m, t) for m, t in pairs if m["independence_violated"] != t["independence_violated"]]
    out["overlap"] = {
        "cells": len(pairs),
        "same_verdict": len(pairs) - len(changed),
        "main_borderline": len(borderline),
        "borderline_become_ci_robust": sum((t["independence_violated_ci"] for _, t in borderline)),
        "changed": [
            {
                "group": m["group"],
                "f": m["injection_fraction"],
                "main": [m["observed_comiss"], m["multiplicative"]],
                "tight": [t["observed_comiss"], t["multiplicative"], t["marginals"]],
            }
            for m, t in changed
        ],
    }
    calls = [
        json.loads(l) for l in gzip.open(RES / "rq_e3_llm_correlation/llm_calls.jsonl.gz", "rt")
    ]
    out["calls_main"] = len(calls)
    out["distinct_request_texts_main"] = len({c["request_text"] for c in calls})
    out["median_latency_s_main"] = median((c["latency_s"] for c in calls))
    with gzip.open(TIGHT / "rq_e3_llm_correlation/llm_calls.jsonl.gz", "rt") as f:
        out["calls_tighten"] = sum((1 for _ in f))
    out["wall_hours_main"] = (
        load_json(RES / "rq_e3_llm_correlation/manifest.json")["wall_seconds"] / 3600
    )
    out["wall_hours_tighten"] = (
        load_json(TIGHT / "rq_e3_llm_correlation/manifest.json")["wall_seconds"] / 3600
    )
    return out


def e4() -> dict:
    rows = load_csv(RES / "rq_e4_baselines/per_config.csv")
    rules = ("independence", "boole", "frechet_joints", "tpag_hw")
    per_rule = {}
    for rule in rules:
        vals = [(r[rule], r["observed"]) for r in rows]
        sound = [(b, o) for b, o in vals if b + EPS >= o]
        nv = [(b, o) for b, o in sound if b < 1 - EPS]
        per_rule[rule] = {
            "coverage": len(sound) / len(vals),
            "unsound": len(vals) - len(sound),
            "nonvacuous_sound": len(nv),
            "mean_slack_sound": mean((min(b, 1.0) - o for b, o in sound))
            if rule != "independence"
            else mean((b - o for b, o in sound)),
            "mean_slack_nonvacuous": mean((b - o for b, o in nv)),
        }
    viol = [r for r in rows if r["observed"] > r["independence"] + EPS]
    by = {r["config"]: r for r in rows}
    return {
        "eta": ETA_E4,
        "n_configs": len(rows),
        "episodes": rows[0]["episodes"],
        "per_rule": per_rule,
        "indep_unsound_configs": [r["config"] for r in viol],
        "indep_unsound_ci95": sum(
            (ci_robust(r["observed"], r["episodes"], r["independence"]) for r in rows)
        ),
        "indep_max_ratio": max((r["observed"] / max(r["independence"], 1e-300) for r in viol)),
        "indep_max_ratio_config": max(
            viol, key=lambda r: r["observed"] / max(r["independence"], 1e-300)
        )["config"],
        "joints_tighten": sum((r["frechet_joints"] < min(r["boole"], 1.0) - EPS for r in rows)),
        "hw_tighten": sum((r["tpag_hw"] < r["frechet_joints"] - EPS for r in rows)),
        "examples": {
            c: {
                k: by[c][k]
                for k in ("observed", "independence", "boole", "frechet_joints", "tpag_hw")
            }
            for c in ("series_n4_c0.3", "parallel_n16_c0.3", "parallel_n32_c0.3", "kofn_3of7_c0.3")
            if c in by
        },
    }


def e5() -> dict:
    m = load_json(RES / "rq_e5_overhead_integration/metrics.json")["metrics"]
    rows = load_csv(RES / "rq_e5_overhead_integration/overhead.csv")
    step = m["llm_step_s"]
    return {
        "overhead": [
            {
                k: r[k]
                for k in (
                    "n_components",
                    "events",
                    "monitor_per_event_us",
                    "monitor_per_episode_ms",
                    "peak_mib",
                )
            }
            for r in rows
        ],
        "llm_step_s": step,
        "ratio_llm_step_over_event_n1024": step / (rows[-1]["monitor_per_event_us"] * 1e-06),
        "ratio_llm_step_over_event_n4": step / (rows[0]["monitor_per_event_us"] * 1e-06),
        "integration": m["integration"],
    }


def e6() -> dict:
    rows = load_csv(RES / "rq_e6_estimation_ablation/per_config.csv")
    nv = [r for r in rows if r["tpag_cp"] < 1 - EPS]
    return {
        "n_configs": len(rows),
        "multi_cut": sum((r["n_cuts"] >= 2 for r in rows)),
        "mean_slack_point": mean((r["tpag_point"] - r["observed"] for r in rows)),
        "mean_slack_cp": mean((r["tpag_cp"] - r["observed"] for r in rows)),
        "nonvacuous_cp": len(nv),
        "mean_slack_point_nonvacuous": mean((r["tpag_point"] - r["observed"] for r in nv)),
        "mean_slack_cp_nonvacuous": mean((r["tpag_cp"] - r["observed"] for r in nv)),
        "hw_active_cp": sum((r["delta_hw_cp"] > EPS for r in rows)),
        "hw_active_point": sum((r["delta_hw_point"] > EPS for r in rows)),
        "hw_retained_fraction": sum((r["delta_hw_cp"] for r in rows))
        / sum((r["delta_hw_point"] for r in rows)),
        "strictly_tighter_than_joint_boole": sum((r["boole_minus_tpag_cp"] > EPS for r in rows)),
        "mean_realized_hw_gain": mean((r["boole_minus_tpag_cp"] for r in rows)),
        "nonvacuous_point": sum((r["tpag_point"] < 1 - EPS for r in rows)),
    }


def llm_shift(pts) -> dict:
    """Freeze each group at fraction zero and test higher injection fractions."""
    by_group: dict[str, list[dict]] = {}
    for p in pts:
        by_group.setdefault(p["group"], []).append(p)
    pairs: list[dict] = []
    for g, ps in by_group.items():
        ps = sorted(ps, key=lambda p: p["injection_fraction"])
        train = ps[0]
        for test in ps[1:]:
            if test["injection_fraction"] <= train["injection_fraction"]:
                continue
            pairs.append(
                {
                    "group": g,
                    "f_train": train["injection_fraction"],
                    "f_test": test["injection_fraction"],
                    "shift": round(test["injection_fraction"] - train["injection_fraction"], 4),
                    "frozen_bound": train["tpag_bound"],
                    "observed_test": test["observed_comiss"],
                    "frozen_holds": bool(train["tpag_bound"] + _TOL >= test["observed_comiss"]),
                    "violation_gap": max(0.0, test["observed_comiss"] - train["tpag_bound"]),
                }
            )
    if not pairs:
        return {"status": "no_shift_pairs"}
    n = len(pairs)
    viol = [p for p in pairs if not p["frozen_holds"]]
    by_shift: dict[str, dict] = {}
    for shift in sorted({p["shift"] for p in pairs}):
        sub = [p for p in pairs if p["shift"] == shift]
        by_shift[f"{shift:g}"] = {
            "n": len(sub),
            "frozen_violation_rate": float(np.mean([not p["frozen_holds"] for p in sub])),
            "mean_violation_gap": float(np.mean([p["violation_gap"] for p in sub])),
        }
    return {
        "status": "measured",
        "n_shift_pairs": n,
        "frozen_certificate_violation_rate": len(viol) / n,
        "max_violation_gap": max((p["violation_gap"] for p in pairs), default=0.0),
        "by_shift": by_shift,
        "pairs": pairs,
    }


def e7() -> dict:
    held = llm_shift(load_json(RES / "rq_e3_llm_correlation/llm_points.json"))
    main = load_json(RES / "rq_e3_llm_correlation/llm_points.json")
    pairs = held["pairs"]
    by_h = defaultdict(lambda: [0, 0])
    ci_viol = 0
    for pr in pairs:
        h = pr["group"].split("|")[2]
        by_h[h][0] += 1
        by_h[h][1] += not pr["frozen_holds"]
        test = find(main, pr["group"], pr["f_test"])
        ci_viol += pr["frozen_bound"] < test["comiss_ci"][0] - EPS
    perm_full = [pr for pr in pairs if "|permissive|" in pr["group"] and pr["f_test"] == 1.0]
    return {
        "llm_pairs": len(pairs),
        "llm_violated": sum((not pr["frozen_holds"] for pr in pairs)),
        "llm_violated_below_test_bootstrap_ci": ci_viol,
        "llm_max_gap": held["max_violation_gap"],
        "llm_by_shift": held["by_shift"],
        "llm_by_hardening": {h: {"pairs": n, "violated": v} for h, (n, v) in by_h.items()},
        "permissive_full_injection": {
            "pairs": len(perm_full),
            "violated": sum((not pr["frozen_holds"] for pr in perm_full)),
        },
    }


def e9() -> dict:
    rows = load_jsonl(RES / "rq_e9_hard_containment/hard_cases.jsonl")
    sched = load_jsonl(RES / "rq_e9_hard_containment/async_scheduling.jsonl")
    m = {
        "hard_cases": rows,
        "mean_scoped_recall": mean(r["scoped_recall"] for r in rows),
        "async_scheduling": {
            r["monitor"]: {k: v for k, v in r.items() if k != "monitor"} for r in sched
        },
    }
    uniq = {c["name"]: c for c in m["hard_cases"]}.values()
    return {
        "unique_cases": [
            {
                k: c[k]
                for k in (
                    "name",
                    "n_nodes",
                    "scoped_precision",
                    "scoped_recall",
                    "scoped_halted_fraction",
                    "global_precision",
                    "global_halted_fraction",
                )
            }
            for c in uniq
        ],
        "unique_mean_recall": mean((c["scoped_recall"] for c in uniq)),
        "unique_min_recall": min((c["scoped_recall"] for c in uniq)),
        "rows_mean_recall": m["mean_scoped_recall"],
        "async": m["async_scheduling"],
    }


def llm_certificate(rows: list[list[int]], eta: float = ETA_CERT) -> float:
    """The E3 certificate of a k-guard stack (rq_e3_llm_correlation._measure_point)."""
    n, k = (len(rows), len(rows[0]))
    names = [f"g{i}" for i in range(k)]
    eps = {names[i]: clopper_pearson_upper(sum((r[i] for r in rows)), n, eta) for i in range(k)}
    cut = frozenset(names)
    k_co = sum((all(r) for r in rows))
    return compute_bound([cut], eps, pair_upper={cut: clopper_pearson_upper(k_co, n, eta)}).b_hw


def _llm_logs(suite: Path) -> list[dict]:
    points = load_json(suite / "rq_e3_llm_correlation/llm_points.json")
    out = []
    with gzip.open(suite / "rq_e3_llm_correlation/llm_calls.jsonl.gz", "rt") as f:
        for p in points:
            k, n = p["k"], p["n_samples"]
            recs = [json.loads(next(f)) for _ in range(n * k)]
            streams = []
            for j in range(0, n * k, k):
                req = recs[j : j + k]
                assert len({r["request_index"] for r in req}) == 1
                assert [r["guard"] for r in req] == [f"g{g}" for g in range(k)]
                assert [r["model"] for r in req] == p["models"]
                if req[0]["request_index"] == 0:
                    streams.append([])
                assert streams and req[0]["request_index"] == len(streams[-1])
                streams[-1].append(
                    {"miss": [int(r["approved_miss"]) for r in req], "text": req[0]["request_text"]}
                )
            expected_stream_size = 300 if n == 900 else (120 if n == 240 else 80)
            assert all(len(stream) == expected_stream_size for stream in streams), p["group"]
            rows = [r["miss"] for stream in streams for r in stream]
            for g in range(k):
                assert abs(sum(r[g] for r in rows) / n - p["marginals"][f"g{g}"]) < EPS, p["group"]
            assert abs(sum(all(r) for r in rows) / n - p["observed_comiss"]) < EPS, p["group"]
            assert abs(llm_certificate(rows) - p["tpag_bound"]) < 1e-12, p["group"]
            out.append({"point": p, "streams": streams, "models": [r["model"] for r in recs[:k]]})
        assert f.readline() == "", "call log longer than its points"
    return out


def _key(p: dict) -> tuple:
    return (p["group"], round(p["injection_fraction"], 6))


def _comiss(rows) -> tuple[int, int]:
    rows = list(rows)
    return (sum((all(r) for r in rows)), len(rows))


def llm_heldout(logs_main: list[dict], logs_tight: list[dict]) -> tuple[dict, list[dict]]:
    """Leave-one-stream-out validity of the LLM certificate on held-out request streams."""

    def splits(logs, suite):
        out = []
        for L in logs:
            S = L["streams"]
            if len(S) < 2:
                continue
            for t in range(len(S)):
                train = [r["miss"] for s, st_ in enumerate(S) if s != t for r in st_]
                kt, nt = _comiss((r["miss"] for r in S[t]))
                p = L["point"]
                out.append(
                    {
                        "suite": suite,
                        "group": p["group"],
                        "f": p["injection_fraction"],
                        "hardening": p["hardening"],
                        "diversity": p["diversity"],
                        "k": p["k"],
                        "held_out_stream": t,
                        "n_train": len(train),
                        "n_test": nt,
                        "certificate": llm_certificate(train),
                        "observed_test": kt / nt,
                        "test_ci95_lo": clopper_pearson_lower(kt, nt, CI95),
                    }
                )
        return out

    def summary(rows):
        nv = [r for r in rows if r["certificate"] < 1 - EPS]
        miss = [r for r in rows if r["certificate"] + EPS < r["observed_test"]]
        return {
            "splits": len(rows),
            "points": len({(r["group"], r["f"]) for r in rows}),
            "n_train": sorted({r["n_train"] for r in rows}),
            "n_test": sorted({r["n_test"] for r in rows}),
            "covered": len(rows) - len(miss),
            "below_test_ci95_lo": sum((r["certificate"] + EPS < r["test_ci95_lo"] for r in rows)),
            "nonvacuous": len(nv),
            "nonvacuous_covered": sum((r["certificate"] + EPS >= r["observed_test"] for r in nv)),
            "median_margin_nonvacuous": median((r["certificate"] - r["observed_test"] for r in nv)),
            "certificates_le_0.10": sum((r["certificate"] <= 0.1 + EPS for r in rows)),
            "misses": [
                {
                    k: r[k]
                    for k in (
                        "group",
                        "f",
                        "held_out_stream",
                        "certificate",
                        "observed_test",
                        "test_ci95_lo",
                    )
                }
                for r in miss
            ],
        }

    s_t, s_m = (splits(logs_tight, "tighten"), splits(logs_main, "main"))
    return ({"tighten": summary(s_t), "main": summary(s_m)}, s_t + s_m)


def llm_composition(logs_main: list[dict]) -> dict:
    """Fig. 1 structure (P1 and P2) or R composed from guards that saw the same requests.

    Stacks of one block, domain, strategy and injection fraction receive the identical
    request stream (base_seed depends only on the stream index and the fraction), so a
    policy pair from one stack and a guard R from another stack give a joint observation
    of a three-guard workflow on every request. Certificates are estimated on one stream
    and tested on the other.
    """
    by = defaultdict(list)
    for L in logs_main:
        p = L["point"]
        by[p["block"], p["domain"], p["strategy"], p["injection_fraction"], p["n_samples"]].append(
            L
        )
    for Ls in by.values():
        ref = [[r["text"] for r in s] for s in Ls[0]["streams"]]
        assert all(([[r["text"] for r in s] for s in L["streams"]] == ref for L in Ls))
    cuts = [frozenset({"P1", "P2"}), frozenset({"R"})]

    def certs(rows):
        n = len(rows)
        kk = [sum((r[i] for r in rows)) for i in range(3)]
        k12 = sum((r[0] & r[1] for r in rows))
        k12r = sum((r[0] & r[1] & r[2] for r in rows))
        k_u = sum((r[0] & r[1] | r[2] for r in rows))
        eps = {
            "P1": clopper_pearson_upper(kk[0], n, ETA_CERT),
            "P2": clopper_pearson_upper(kk[1], n, ETA_CERT),
            "R": clopper_pearson_upper(kk[2], n, ETA_CERT),
        }
        pair = {cuts[0]: clopper_pearson_upper(k12, n, ETA_CERT)}
        lo = clopper_pearson_lower(k12r, n, ETA_CERT)
        return {
            "boole_frechet": compute_bound(cuts, eps, pair_upper=pair).b_hw,
            "tpag_hw": compute_bound(
                cuts, eps, pair_upper=pair, cut_joint_lower=lambda a, b: lo
            ).b_hw,
            "direct_end_to_end": clopper_pearson_upper(k_u, n, ETA_CERT),
        }

    tests, combos, streams_used = ([], 0, set())
    policy_models, r_models = (set(), set())
    for key, Ls in sorted(by.items()):
        if len(Ls[0]["streams"]) < 2:
            continue
        for hard in ("permissive", "strict"):
            stacks = [L for L in Ls if L["point"]["hardening"] == hard and L["point"]["k"] == 2]
            for A in stacks:
                for B in stacks:
                    if B is A:
                        continue
                    for g in range(B["point"]["k"]):
                        combos += 1
                        policy_models.add(tuple(A["models"]))
                        r_models.add(B["models"][g])
                        for tr in range(2):
                            te = 1 - tr
                            streams_used.add((key, tr))

                            def rows(s):
                                return [
                                    (
                                        A["streams"][s][i]["miss"][0],
                                        A["streams"][s][i]["miss"][1],
                                        B["streams"][s][i]["miss"][g],
                                    )
                                    for i in range(len(A["streams"][s]))
                                ]

                            c = certs(rows(tr))
                            test = rows(te)
                            k_u = sum((r[0] & r[1] | r[2] for r in test))
                            tests.append(
                                {
                                    "hardening": hard,
                                    "f": key[3],
                                    **c,
                                    "observed_test": k_u / len(test),
                                    "test_ci95_lo": clopper_pearson_lower(k_u, len(test), CI95),
                                }
                            )

    def summary(ts):
        out = {"tests": len(ts)}
        for rule in ("boole_frechet", "tpag_hw", "direct_end_to_end"):
            out[rule] = {
                "covered": sum((t[rule] + EPS >= t["observed_test"] for t in ts)),
                "below_test_ci95_lo": sum((t[rule] + EPS < t["test_ci95_lo"] for t in ts)),
                "nonvacuous": sum((t[rule] < 1 - EPS for t in ts)),
                "median": median((t[rule] for t in ts)),
                "nonvacuous_covered": sum(
                    (t[rule] < 1 - EPS and t[rule] + EPS >= t["observed_test"] for t in ts)
                ),
            }
        gain = [
            t["boole_frechet"] - t["tpag_hw"] for t in ts if t["tpag_hw"] < t["boole_frechet"] - EPS
        ]
        out["hw_tightens"] = len(gain)
        out["hw_mean_gain_when_active"] = mean(gain)
        out["mean_excess_tpag_over_direct"] = mean(
            (t["tpag_hw"] - t["direct_end_to_end"] for t in ts)
        )
        return out

    return {
        "structure": "Series(Parallel(P1,P2), R) on shared request streams",
        "combinations": combos,
        "confidence": {
            "boole_frechet": 1 - 4 * ETA_CERT,
            "tpag_hw": 1 - 5 * ETA_CERT,
            "direct_end_to_end": 1 - ETA_CERT,
        },
        "all": summary(tests),
        "by_hardening": {
            h: summary([t for t in tests if t["hardening"] == h]) for h in ("permissive", "strict")
        },
        "provenance": {
            "policy_pair_models": len(policy_models),
            "same_model_policy_pairs": sum((len(set(m)) == 1 for m in policy_models)),
            "r_models": len(r_models),
            "request_streams": len(streams_used),
            "requests_per_stream": sorted(
                {
                    len(s)
                    for Ls in by.values()
                    for L in Ls
                    for s in L["streams"]
                    if len(Ls[0]["streams"]) >= 2
                }
            ),
        },
    }


def llm_replay(logs_main: list[dict], logs_tight: list[dict]) -> dict:
    """Requests that the re-measurement replays with the same text and per-call seed.

    Both runs use master seed 20260607, the stream seed depends only on the stream
    index and the fraction, make_dataset draws requests one at a time, and the per-call
    seed is base_seed + 131 r + g. The first requests of a re-measured stream therefore
    replay the main-suite stream with the same index.
    """
    midx = {_key(L["point"]): L for L in logs_main}
    per_model = defaultdict(lambda: [0, 0])
    reqs = rows_same = dec_same = dec_tot = 0
    cells, cell_rows = ([], [])
    for T in logs_tight:
        M = midx.get(_key(T["point"]))
        if M is None:
            continue
        assert M["models"] == T["models"]
        n_rep = 0
        for s in range(min(len(M["streams"]), len(T["streams"]))):
            for i, a in enumerate(M["streams"][s]):
                b = T["streams"][s][i]
                assert a["text"] == b["text"]
                n_rep += 1
                reqs += 1
                rows_same += a["miss"] == b["miss"]
                for g, (x, y) in enumerate(zip(a["miss"], b["miss"])):
                    dec_tot += 1
                    dec_same += x == y
                    per_model[T["models"][g]][0] += x == y
                    per_model[T["models"][g]][1] += 1
        fresh = [
            r["miss"]
            for s, st_ in enumerate(T["streams"])
            for i, r in enumerate(st_)
            if s >= len(M["streams"]) or i >= len(M["streams"][s])
        ]
        kf, nf = _comiss(fresh)
        cells.append(
            {
                "group": T["point"]["group"],
                "f": T["point"]["injection_fraction"],
                "replayed_requests": n_rep,
                "fresh_requests": nf,
                "main_certificate": M["point"]["tpag_bound"],
                "tight_observed_all": T["point"]["observed_comiss"],
                "tight_ci95_lo_all": clopper_pearson_lower(
                    round(T["point"]["observed_comiss"] * T["point"]["n_samples"]),
                    T["point"]["n_samples"],
                    CI95,
                ),
                "tight_observed_fresh": kf / nf,
                "tight_ci95_lo_fresh": clopper_pearson_lower(kf, nf, CI95),
                "main_observed": M["point"]["observed_comiss"],
                "tight_certificate": T["point"]["tpag_bound"],
            }
        )
        cell_rows.append((T["point"]["group"], T["point"]["injection_fraction"], M, T))
    worst = None
    for grp, f, M, T in cell_rows:
        for s in range(min(len(M["streams"]), len(T["streams"]))):
            a = M["streams"][s]
            b = T["streams"][s][: len(a)]
            ca, cb = (
                _comiss((r["miss"] for r in a))[0] / len(a),
                _comiss((r["miss"] for r in b))[0] / len(a),
            )
            if worst is None or abs(ca - cb) > abs(worst["main"] - worst["tight"]):
                worst = {
                    "group": grp,
                    "f": f,
                    "stream": s,
                    "requests": len(a),
                    "main": ca,
                    "tight": cb,
                }
    pm = {m: a / t for m, (a, t) in per_model.items()}
    return {
        "cells": len(cells),
        "replayed_requests": reqs,
        "identical_decision_rows": rows_same / reqs,
        "identical_guard_decisions": dec_same / dec_tot,
        "per_model_agreement": pm,
        "largest_replay_comiss_change": worst,
        "cross_run": {
            "main_cert_covers_tight_all": sum(
                (c["main_certificate"] + EPS >= c["tight_observed_all"] for c in cells)
            ),
            "main_cert_below_tight_ci95_lo_all": sum(
                (c["main_certificate"] + EPS < c["tight_ci95_lo_all"] for c in cells)
            ),
            "main_cert_covers_tight_fresh": sum(
                (c["main_certificate"] + EPS >= c["tight_observed_fresh"] for c in cells)
            ),
            "main_cert_below_tight_ci95_lo_fresh": sum(
                (c["main_certificate"] + EPS < c["tight_ci95_lo_fresh"] for c in cells)
            ),
            "tight_cert_covers_main": sum(
                (c["tight_certificate"] + EPS >= c["main_observed"] for c in cells)
            ),
            "misses_fresh": [
                c for c in cells if c["main_certificate"] + EPS < c["tight_observed_fresh"]
            ],
        },
    }


def llm_fair_product() -> dict:
    """Independence comparators: point-estimate product vs product of CP upper marginals."""
    out = {}
    for suite, root in (("main", RES), ("tighten", TIGHT)):
        pts = load_json(root / "rq_e3_llm_correlation/llm_points.json")
        strict, robust, ratios = (0, 0, [])
        for p in pts:
            n = p["n_samples"]
            prod = math.prod(
                (clopper_pearson_upper(round(v * n), n, ETA_CERT) for v in p["marginals"].values())
            )
            if p["observed_comiss"] > prod + EPS:
                strict += 1
                robust += p["comiss_ci"][0] > prod + EPS
                ratios.append(p["observed_comiss"] / prod)
        point = [
            p["underreport_ratio"]
            for p in pts
            if p["independence_violated_ci"] and p["underreport_ratio"]
        ]
        out[suite] = {
            "cp_upper_product": {
                "violated": strict,
                "violated_ci": robust,
                "max_ratio": max(ratios, default=None),
                "median_ratio": median(ratios),
            },
            "point_product_ci_robust_ratios": {
                "n": len(point),
                "median": median(point),
                "below_5x": sum((r < 5 for r in point)),
                "max": max(point),
            },
        }
    return out


def llm_mixture() -> dict:
    """max(B_0, B_1) bounds every stationary injection fraction: P_f = (1-f) P_0 + f P_1."""
    out = {}
    for suite, root in (("main", RES), ("tighten", TIGHT)):
        g = defaultdict(dict)
        for p in load_json(root / "rq_e3_llm_correlation/llm_points.json"):
            g[p["group"]][round(p["injection_fraction"], 6)] = p
        tests = cov = nv = 0
        resid = []
        for d in g.values():
            if 0.0 not in d or 1.0 not in d:
                continue
            b = max(d[0.0]["tpag_bound"], d[1.0]["tpag_bound"])
            for f, p in d.items():
                if 0 < f < 1:
                    tests += 1
                    cov += b + EPS >= p["observed_comiss"]
                    nv += b < 1 - EPS
                    resid.append(
                        p["observed_comiss"]
                        - ((1 - f) * d[0.0]["observed_comiss"] + f * d[1.0]["observed_comiss"])
                    )
        out[suite] = {
            "intermediate_points": tests,
            "covered": cov,
            "nonvacuous": nv,
            "interpolation_residual": [min(resid), max(resid)],
        }
    return out


def llm_floors() -> dict:
    """Zero-failure Clopper-Pearson floors and how many certificates sit at them."""
    floors = {}
    for m in (80, 240, 900):
        f = 1 - ETA_CERT ** (1 / m)
        assert abs(f - clopper_pearson_upper(0, m, ETA_CERT)) < 1e-12
        floors[str(m)] = {"floor": f, "ln_bound": math.log(1 / ETA_CERT) / m}
    out = {
        "eta": ETA_CERT,
        "floors": floors,
        "episodes_to_certify": {
            str(d): math.ceil(math.log(ETA_CERT) / math.log(1 - d)) for d in (0.01, 0.001, 0.0001)
        },
        "text_level_floors": {str(m): 1 - ETA_CERT ** (1 / m) for m in (10, 30, 40)},
    }
    for suite, root in (("main", RES), ("tighten", TIGHT)):
        pts = load_json(root / "rq_e3_llm_correlation/llm_points.json")
        le10 = [p for p in pts if p["tpag_bound"] <= 0.1 + EPS]
        le05 = [p for p in pts if p["tpag_bound"] <= 0.05 + EPS]
        pos = [p for p in pts if p["observed_comiss"] > 0]
        hc = [p for p in pts if p["diversity"] == "cross_family" and p["hardening"] == "strict"]
        out[suite] = {
            "le_0.10": len(le10),
            "le_0.10_zero_comiss": sum((p["observed_comiss"] == 0 for p in le10)),
            "le_0.05": len(le05),
            "le_0.05_zero_comiss": sum((p["observed_comiss"] == 0 for p in le05)),
            "with_comiss": len(pos),
            "with_comiss_le_0.10": sum((p["tpag_bound"] <= 0.1 + EPS for p in pos)),
            "hardened_cross_family": {
                "points": len(hc),
                "zero_comiss": sum((p["observed_comiss"] == 0 for p in hc)),
                "zero_comiss_by_n": {
                    str(n): sum((p["observed_comiss"] == 0 and p["n_samples"] == n for p in hc))
                    for n in sorted({p["n_samples"] for p in hc})
                },
            },
            "at_floor": {
                str(n): sum((p["observed_comiss"] == 0 and p["n_samples"] == n for p in pts))
                for n in sorted({p["n_samples"] for p in pts})
            },
            "at_floor_by_design": {
                f"{d}|{h}|k{k}": sum(
                    (
                        p["observed_comiss"] == 0
                        for p in pts
                        if (p["diversity"], p["hardening"], p["k"]) == (d, h, k)
                    )
                )
                for d, h, k in sorted({(p["diversity"], p["hardening"], p["k"]) for p in pts})
            },
            "zero_comiss_at_every_fraction": sorted(
                (
                    g
                    for g in {p["group"] for p in pts}
                    if all((p["observed_comiss"] == 0 for p in pts if p["group"] == g))
                )
            ),
        }
    return out


def llm_pool(logs_main: list[dict]) -> dict:
    """Distinct request texts per point and agreement of repeated calls on one request."""
    distinct = defaultdict(list)
    agree = total = 0
    per_model = defaultdict(lambda: [0, 0])
    for L in logs_main:
        cells = defaultdict(list)
        texts = set()
        for s in L["streams"]:
            for r in s:
                texts.add(r["text"])
                for g, x in enumerate(r["miss"]):
                    cells[g, r["text"]].append(x)
        distinct[str(L["point"]["injection_fraction"])].append(len(texts))
        for (g, _), v in cells.items():
            a = len(v) if len(set(v)) == 1 else 0
            agree += a
            total += len(v)
            per_model[L["models"][g]][0] += a
            per_model[L["models"][g]][1] += len(v)
    return {
        "distinct_request_texts": {
            f: {"min": min(v), "median": median(v), "max": max(v)}
            for f, v in sorted(distinct.items())
        },
        "calls_in_unanimous_request_cells": agree / total,
        "per_model": {m: a / t for m, (a, t) in sorted(per_model.items())},
    }


def llm_strata() -> dict:
    """Descriptive tallies of the committed LLM points by design stratum (no new analysis)."""

    def counts(ps):
        return {
            "points": len(ps),
            "viol": sum((p["independence_violated"] for p in ps)),
            "viol_ci": sum((p["independence_violated_ci"] for p in ps)),
        }

    def stratum(ps):
        fr = sorted({p["injection_fraction"] for p in ps})
        return {
            **counts(ps),
            "by_fraction": {
                str(f): counts([p for p in ps if abs(p["injection_fraction"] - f) < EPS])
                for f in fr
            },
            "comiss": mean((p["observed_comiss"] for p in ps)),
            "product": mean((p["multiplicative"] for p in ps)),
            "median_bound": median((p["tpag_bound"] for p in ps)),
            "zero_comiss": sum((p["observed_comiss"] == 0 for p in ps)),
        }

    main_groups = {p["group"] for p in load_json(RES / "rq_e3_llm_correlation/llm_points.json")}
    out = {}
    for suite, root in (("main", RES), ("tighten", TIGHT)):
        pts = load_json(root / "rq_e3_llm_correlation/llm_points.json")
        strata = {
            f"{d}|{h}": stratum([p for p in pts if (p["diversity"], p["hardening"]) == (d, h)])
            for d, h in sorted({(p["diversity"], p["hardening"]) for p in pts})
        }
        clean = [p for p in pts if p["injection_fraction"] == 0 and p["independence_violated"]]
        large = [p for p in pts if p["size_tier"] == "large"]
        same_ci = [
            p for p in pts if p["diversity"] == "same_model" and p["independence_violated_ci"]
        ]
        hard = [p for p in pts if p["hardening"] == "strict"]
        perm = [p for p in pts if p["hardening"] == "permissive"]
        worst = max(
            (p for p in pts if p["independence_violated"]), key=lambda p: p["underreport_ratio"]
        )
        n, k = (worst["n_samples"], worst["k"])
        c = round(worst["observed_comiss"] * n)
        groups = sorted({p["group"] for p in pts})
        new = [g for g in groups if g not in main_groups]
        new_pts = [p for p in pts if p["group"] in new]
        out[suite] = {
            "strata": strata,
            "all": stratum(pts),
            "clean_input_violations": {
                "viol": len(clean),
                "viol_ci": sum((p["independence_violated_ci"] for p in clean)),
                "by_diversity": {
                    d: sum((p["diversity"] == d for p in clean))
                    for d in sorted({p["diversity"] for p in clean})
                },
            },
            "large_tier": {
                "points": len(large),
                "designs": sorted(
                    {f"{p['diversity']}|{p['hardening']}|n{p['n_samples']}" for p in large}
                ),
            },
            "same_model_ci_robust": {
                "points": len(same_ci),
                "comiss_equals_min_marginal": sum(
                    (
                        abs(p["observed_comiss"] - min(p["marginals"].values())) < EPS
                        for p in same_ci
                    )
                ),
            },
            "hardened": {
                **counts(hard),
                "viol_by_diversity": {
                    d: sum((p["independence_violated"] for p in hard if p["diversity"] == d))
                    for d in sorted({p["diversity"] for p in hard})
                },
            },
            "permissive": counts(perm),
            "largest_ratio_identity": {
                "group": worst["group"],
                "injection_fraction": worst["injection_fraction"],
                "n": n,
                "k": k,
                "ratio": worst["underreport_ratio"],
                "comiss_count": c,
                "marginal_counts": sorted((round(v * n) for v in worst["marginals"].values())),
                "one_over_eps_pow_k_minus_1": (n / c) ** (k - 1),
            },
            "groups": {
                "total": len(groups),
                "repeat_main_suite": len(groups) - len(new),
                "new": len(new),
                "new_points": len(new_pts),
                "new_viol": sum((p["independence_violated"] for p in new_pts)),
                "new_groups": new if suite == "tighten" else [],
            },
        }
    return out


def llm_parse() -> dict:
    """Every logged guard response is a JSON decision that matches the recorded miss."""
    out = {}
    for suite, root in (("main", RES), ("tighten", TIGHT)):
        calls = parsed = matches = 0
        texts, by_domain = (set(), defaultdict(set))
        points = load_json(root / "rq_e3_llm_correlation/llm_points.json")
        with gzip.open(root / "rq_e3_llm_correlation/llm_calls.jsonl.gz", "rt") as f:
            for p in points:
                for _ in range(p["n_samples"] * p["k"]):
                    r = json.loads(next(f))
                    calls += 1
                    try:
                        d = json.loads(r["response"]).get("decision")
                    except (ValueError, AttributeError):
                        d = None
                    parsed += d in ("APPROVE", "BLOCK")
                    matches += (d == "APPROVE") == (r["approved_miss"] in (True, "True"))
                    texts.add(r["request_text"])
                    by_domain[p["domain"]].add(r["request_text"])
        out[suite] = {
            "calls": calls,
            "json_decisions": parsed,
            "decision_matches_miss": matches,
            "distinct_request_texts": len(texts),
            "distinct_request_texts_by_domain": {d: len(t) for d, t in sorted(by_domain.items())},
        }
    return out


def e2_fullcut() -> dict:
    """Parallel banks certified from the full-cut counter alone (N_est = 1).

    For a parallel bank the system event is the full cut, so its committed observed
    rate and episode count give the Clopper-Pearson limit of the one quantity that
    enters the bound; registering only it keeps the confidence at 1 - eta.
    """
    rows = [r for r in load_csv(RES / "rq_e2_topology/per_config.csv") if r["family"] == "parallel"]
    out = []
    for r in sorted(rows, key=lambda r: (r["config"].endswith("c0.3"), r["n_components"])):
        n, m = (int(r["n_components"]), int(r["episodes"]))
        out.append(
            {
                "config": r["config"],
                "n": n,
                "episodes": m,
                "observed": r["observed"],
                "b_hw_all_pairs": r["b_hw"],
                "confidence_all_pairs": max(0.0, 1 - n_est_pairs(n) * ETA_CERT),
                "fullcut_cp": clopper_pearson_upper(round(r["observed"] * m), m, ETA_CERT),
                "confidence_fullcut": 1 - ETA_CERT,
            }
        )
    others = [
        r
        for r in load_csv(RES / "rq_e2_topology/per_config.csv")
        if r["family"] != "parallel" and r["b_hw"] < 1 - EPS
    ]
    return {
        "rows": out,
        "parallel_summary": {
            "configs": len(out),
            "nonvacuous": sum((r["fullcut_cp"] < 1 - EPS for r in out)),
            "mean_slack": mean((r["fullcut_cp"] - r["observed"] for r in out)),
            "confidence": 1 - ETA_CERT,
        },
        "nonparallel_nonvacuous": {
            "configs": len(others),
            "max_n": max((int(r["n_components"]) for r in others)),
            "min_confidence_all_pairs": min(
                (max(0.0, 1 - n_est_pairs(int(r["n_components"])) * ETA_CERT) for r in others)
            ),
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force and any(
        p.exists() for p in (OUT, SPLITS_OUT, OUT.parent / "llm_held_out_regime.json")
    ):
        parser.error("derived outputs exist; pass --force to replace them")
    stats = {
        "e1": e1(),
        "e2": e2(),
        "e3": e3(),
        "e4": e4(),
        "e5": e5(),
        "e6": e6(),
        "e7": e7(),
        "e9": e9(),
    }
    logs_main, logs_tight = (_llm_logs(RES), _llm_logs(TIGHT))
    heldout, split_rows = llm_heldout(logs_main, logs_tight)
    cells = {_key(L["point"]) for L in logs_main} | {_key(L["point"]) for L in logs_tight}
    stats.update(
        {
            "llm_distinct_cells": len(cells),
            "llm_models": {
                "main": sorted({m for L in logs_main for m in L["models"]}),
                "tighten": sorted({m for L in logs_tight for m in L["models"]}),
            },
            "llm_heldout": heldout,
            "llm_composition": llm_composition(logs_main),
            "llm_replay": llm_replay(logs_main, logs_tight),
            "llm_fair_product": llm_fair_product(),
            "llm_mixture": llm_mixture(),
            "llm_floors": llm_floors(),
            "llm_pool": llm_pool(logs_main),
            "llm_pool_tighten": llm_pool(logs_tight),
            "llm_heldout_total": {
                k: heldout["main"][k] + heldout["tighten"][k]
                for k in ("splits", "covered", "below_test_ci95_lo")
            },
            "e2_fullcut": e2_fullcut(),
            "llm_strata": llm_strata(),
            "llm_parse": llm_parse(),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "llm_held_out_regime.json").write_text(
        json.dumps(llm_shift(load_json(RES / "rq_e3_llm_correlation/llm_points.json")), indent=1)
        + "\n"
    )
    OUT.write_text(json.dumps(stats, indent=1, default=float) + "\n")
    SPLITS_OUT.write_text(json.dumps(split_rows, indent=1) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} and held-out/shift rows")


if __name__ == "__main__":
    main()
