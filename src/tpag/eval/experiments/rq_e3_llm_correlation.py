"""Rq e3 llm correlation for the TPAG replication package."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ...adapters.llm import OllamaClient
from ...agents.guards import PERMISSIVE_SYSTEM_PROMPT, POLICY_SYSTEM_PROMPT, LLMGuard
from ...bound import compute_bound
from ...estimation import clopper_pearson_upper
from ...sim.attacks import make_dataset
from ...sim.domains.code_review import PROMPTS as CODE_REVIEW_PROMPTS
from .. import model_zoo
from ..bound_values import summarize_bound_values
from ..config import ExpConfig
from ..llm_runner import run_guards_over_dataset
from ..stats import bootstrap_ci
from . import finalize, jsonl_appender, make_recorder

RQ = "RQ-E3"


def recorded_groups(cfg: ExpConfig) -> list[dict]:
    """Freeze the reported design instead of selecting different available models."""
    from ..config import REPO_ROOT

    suites = {"exec_e3_final.yaml": "results", "exec_e3_tighten.yaml": "results_tighten"}
    name = Path(cfg.source).name
    if name not in suites or cfg.tier.name != "full":
        raise ValueError("E3 requires a reported full-tier configuration")
    points = json.loads(
        (REPO_ROOT / suites[name] / "rq_e3_llm_correlation/llm_points.json").read_text()
    )
    blocks = {block["name"]: block for block in cfg.blocks}
    groups = {}
    for point in points:
        if point["group"] not in groups:
            block = blocks[point["block"]]
            if block["n_requests"] * block["n_seeds"] != point["n_samples"]:
                raise ValueError("Configuration changes the reported sample size")
            groups[point["group"]] = {
                "condition": {
                    "name": point["group"].split("|")[0],
                    "models": point["models"],
                    "k": point["k"],
                    "params_b": point["params_b"],
                    "mean_params_b": point["mean_params_b"],
                    "max_params_b": max(point["params_b"]),
                    "size_tier": point["size_tier"],
                    "diversity": point["diversity"],
                },
                "domain": point["domain"],
                "hardening": point["hardening"],
                "strategy": point["strategy"],
                "block": point["block"],
                "n_requests": block["n_requests"],
                "n_seeds": block["n_seeds"],
                "injection_fractions": [],
            }
        groups[point["group"]]["injection_fractions"].append(point["injection_fraction"])
    return list(groups.values())


def _prompt_for(domain: str, hardening: str) -> str:
    if domain == "code_review":
        return CODE_REVIEW_PROMPTS[hardening]
    return PERMISSIVE_SYSTEM_PROMPT if hardening == "permissive" else POLICY_SYSTEM_PROMPT


def _build_guards(condition: dict, domain: str, hardening: str, url: str) -> list[LLMGuard]:
    prompt = _prompt_for(domain, hardening)
    guards = []
    for gi, tag in enumerate(condition["models"]):
        client = OllamaClient(model=tag, base_url=url, temperature=0.0)
        guards.append(LLMGuard(f"g{gi}", client, prompt))
    return guards


def _measure_point(group, af, cfg, rec) -> dict:
    cond = group["condition"]
    guards = _build_guards(cond, group["domain"], group["hardening"], cfg.ollama_url)
    gnames = [g.name for g in guards]
    cut = frozenset(gnames)
    eta = cfg.tier.eta_grid[0]
    n_req = int(group.get("n_requests", cfg.tier.n_requests))
    n_seeds = int(group.get("n_seeds", cfg.tier.n_seeds))
    stacks, total_lat, total_calls = ([], 0.0, 0)
    for s in range(n_seeds):
        base_seed = cfg.seed + 1009 * s + int(af * 100)
        rng = np.random.default_rng(base_seed)
        dataset = make_dataset(
            n_req,
            attack_fraction=af,
            rng=rng,
            domain=group["domain"],
            strategy=group["strategy"],
            guard_count=cond["k"],
        )
        res = run_guards_over_dataset(guards, dataset, recorder=rec, seed=base_seed)
        stacks.append(res.miss_matrix)
        total_lat += res.latency_s_sum
        total_calls += res.n_calls
    miss = np.vstack(stacks)
    N = miss.shape[0]
    k_i = {g: int(miss[:, i].sum()) for i, g in enumerate(gnames)}
    co = np.all(miss == 1, axis=1).astype(float)
    k_co = int(co.sum())
    marg = {g: k_i[g] / N for g in gnames}
    marg_ci = {
        g: bootstrap_ci(miss[:, i], n_boot=cfg.tier.boot_resamples)[1:]
        for i, g in enumerate(gnames)
    }
    comiss = k_co / N
    _, co_lo, co_hi = bootstrap_ci(co, n_boot=cfg.tier.boot_resamples)
    eps_upper = {g: clopper_pearson_upper(k_i[g], N, eta) for g in gnames}
    pair_upper = {cut: clopper_pearson_upper(k_co, N, eta)}
    bound = compute_bound([cut], eps_upper, pair_upper=pair_upper).b_hw
    mult = float(np.prod([marg[g] for g in gnames]))
    return {
        "group": f"{cond['name']}|{group['domain']}|{group['hardening']}|{group['strategy']}",
        "block": group.get("block", "default"),
        "diversity": cond["diversity"],
        "k": cond["k"],
        "models": cond["models"],
        "params_b": cond["params_b"],
        "size_tier": cond["size_tier"],
        "mean_params_b": cond["mean_params_b"],
        "domain": group["domain"],
        "hardening": group["hardening"],
        "strategy": group["strategy"],
        "injection_fraction": af,
        "n_samples": N,
        "marginals": marg,
        "marginal_ci": marg_ci,
        "observed_comiss": comiss,
        "comiss_ci": [co_lo, co_hi],
        "multiplicative": mult,
        "tpag_bound": bound,
        "independence_violated": bool(comiss > mult + 1e-09),
        "independence_violated_ci": bool(co_lo > mult + 1e-09),
        "tpag_holds": bool(bound + 1e-09 >= comiss),
        "underreport_ratio": comiss / mult if mult > 1e-09 else None,
        "mean_latency_s": total_lat / total_calls if total_calls else None,
        "confidence": 1 - (len(gnames) + 1) * eta,
    }


def _size_efficiency(points: list[dict]) -> dict:
    """Per-size-tier summary: does the phenomenon appear with smaller models, and at
    what parameter/latency cost?"""
    out: dict[str, dict] = {}
    for tier in ("small", "mid", "large"):
        pts = [p for p in points if p["size_tier"] == tier]
        if not pts:
            continue
        viol = [p for p in pts if p["independence_violated"]]
        lats = [p["mean_latency_s"] for p in pts if p["mean_latency_s"] is not None]
        out[tier] = {
            "n_points": len(pts),
            "mean_params_b": float(np.mean([p["mean_params_b"] for p in pts])),
            "mean_marginal_miss": float(
                np.mean([np.mean(list(p["marginals"].values())) for p in pts])
            ),
            "mean_comiss": float(np.mean([p["observed_comiss"] for p in pts])),
            "independence_violated_rate": len(viol) / len(pts),
            "tpag_holds_rate": float(np.mean([p["tpag_holds"] for p in pts])),
            "mean_latency_s": float(np.mean(lats)) if lats else None,
        }
    tiers_violating = [t for t, v in out.items() if v["independence_violated_rate"] > 0]
    order = {"small": 0, "mid": 1, "large": 2}
    out["smallest_tier_violating_independence"] = (
        min(tiers_violating, key=lambda t: order[t]) if tiers_violating else None
    )
    return out


def _axis_breakdown(points: list[dict], key: str) -> dict:
    """Aggregate the deferred-axis values (domain / strategy / diversity / k / block):
    how often independence is violated (point-estimate and CI-robust) and whether
    TPAG holds, per value of ``key``."""
    out: dict[str, dict] = {}
    for val in sorted({str(p[key]) for p in points}):
        pts = [p for p in points if str(p[key]) == val]
        lats = [p["mean_latency_s"] for p in pts if p["mean_latency_s"] is not None]
        bounds = [p["tpag_bound"] for p in pts if p.get("tpag_bound") is not None]
        out[val] = {
            "n_points": len(pts),
            "mean_comiss": float(np.mean([p["observed_comiss"] for p in pts])),
            "mean_multiplicative": float(np.mean([p["multiplicative"] for p in pts])),
            "independence_violated_rate": float(np.mean([p["independence_violated"] for p in pts])),
            "independence_violated_ci_rate": float(
                np.mean([p["independence_violated_ci"] for p in pts])
            ),
            "tpag_holds_rate": float(np.mean([p["tpag_holds"] for p in pts])),
            "median_bound": float(np.median(bounds)) if bounds else None,
            "frac_bound_le_0.10": float(np.mean([b <= 0.1 for b in bounds])) if bounds else None,
            "mean_latency_s": float(np.mean(lats)) if lats else None,
        }
    return out


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e3_llm_correlation", argv=argv, force=force)
    if not use_llm:
        return finalize(rec, cfg, RQ, "skipped_no_llm", {"reason": "run invoked with --no-llm"})
    status = model_zoo.probe_zoo(cfg)
    if not status.ollama_up:
        rec.write_json(
            "missing_models.json", {"reason": status.reason, "pull": model_zoo.pull_hints(status)}
        )
        return finalize(
            rec,
            cfg,
            RQ,
            "skipped_no_ollama",
            {"reason": status.reason, "pull_hints": model_zoo.pull_hints(status)},
        )
    groups = recorded_groups(cfg)
    required = {tag for group in groups for tag in group["condition"]["models"]}
    missing = required - {model.tag for model in status.available}
    if missing:
        return finalize(rec, cfg, RQ, "skipped_missing_models", {"missing": sorted(missing)})
    rec.set("serving_model_parameters", {model.tag: model.params_b for model in status.available})
    (rec.run_dir / "llm_calls.jsonl").unlink(missing_ok=True)
    append_point = jsonl_appender(rec, "llm_points.jsonl")
    append_error = jsonl_appender(rec, "errors.jsonl")
    points: list[dict] = []
    n_errors = 0
    for gi, group in enumerate(groups):
        print(
            f"  [RQ-E3] group {gi + 1}/{len(groups)} [{group['condition']['size_tier']}|{group.get('block', 'default')}]: {group['condition']['name']} | {group['domain']} | {group['hardening']} | {group['strategy']}"
        )
        for af in group.get("injection_fractions", cfg.tier.injection_fractions):
            try:
                pt = _measure_point(group, af, cfg, rec)
                points.append(pt)
                append_point(pt)
            except Exception as exc:
                n_errors += 1
                append_error(
                    {
                        "group": group["condition"]["name"],
                        "af": af,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if not points:
        return finalize(
            rec,
            cfg,
            RQ,
            "skipped_missing_models",
            {"reason": "no operating points produced", "n_errors": n_errors},
        )
    inter = [p for p in points if 0.0 < p["injection_fraction"] < 1.0]
    same = [p for p in points if p["diversity"] == "same_model"]
    cross = [p for p in points if p["diversity"] == "cross_family"]
    ratios = [
        p["underreport_ratio"]
        for p in points
        if p["underreport_ratio"] is not None and p["independence_violated"]
    ]
    axis_breakdowns = {
        "by_domain": _axis_breakdown(points, "domain"),
        "by_strategy": _axis_breakdown(points, "strategy"),
        "by_diversity": _axis_breakdown(points, "diversity"),
        "by_hardening": _axis_breakdown(points, "hardening"),
        "by_k": _axis_breakdown(points, "k"),
        "by_block": _axis_breakdown(points, "block"),
    }
    bound_value_summary = summarize_bound_values(points)
    metrics = {
        "n_points": len(points),
        "n_groups": len(groups),
        "design": "blocks" if cfg.blocks else "cartesian",
        "blocks": [b.get("name", f"block{i}") for i, b in enumerate(cfg.blocks)],
        "available_eligible": len(status.available),
        "models_by_tier": {t: [m.tag for m in ms] for t, ms in status.by_tier(cfg).items()},
        "excluded_oversized": [m.tag for m in status.excluded_oversized],
        "axes_covered": {
            "domains": sorted({p["domain"] for p in points}),
            "strategies": sorted({p["strategy"] for p in points}),
            "diversity": sorted({p["diversity"] for p in points}),
            "ks": sorted({p["k"] for p in points}),
        },
        "independence_violated_points": int(sum((p["independence_violated"] for p in points))),
        "independence_violated_intermediate": int(sum((p["independence_violated"] for p in inter))),
        "independence_violated_ci_robust": int(
            sum((p["independence_violated_ci"] for p in points))
        ),
        "tpag_holds_points": int(sum((p["tpag_holds"] for p in points))),
        "tpag_holds_all": bool(all((p["tpag_holds"] for p in points))),
        "max_underreport_ratio": max(ratios) if ratios else None,
        "mean_comiss_same_model": float(np.mean([p["observed_comiss"] for p in same]))
        if same
        else None,
        "mean_comiss_cross_family": float(np.mean([p["observed_comiss"] for p in cross]))
        if cross
        else None,
        "size_efficiency": _size_efficiency(points),
        "axis_breakdowns": axis_breakdowns,
        "bound_value_summary": bound_value_summary,
        "n_errors": n_errors,
    }
    rec.write_json("llm_points.json", points)
    return finalize(
        rec,
        cfg,
        RQ,
        "measured",
        metrics,
        provenance={
            "points": "llm_points.json (+ streamed llm_points.jsonl)",
            "llm_calls": "llm_calls.jsonl",
            "model_tags": [m.tag for m in status.available],
        },
    )
