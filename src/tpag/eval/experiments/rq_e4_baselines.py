"""Rq e4 baselines for the TPAG replication package."""

from __future__ import annotations
from ...sim.topology_gen import soundness_suite
from .. import comparison
from ..config import ExpConfig
from . import finalize, make_recorder

RQ = "RQ-E4"


def run(cfg: ExpConfig, use_llm: bool = True, *, argv=None, force: bool = False) -> dict:
    rec = make_recorder(cfg, RQ, "rq_e4_baselines", argv=argv, force=force)
    t = cfg.tier
    sizes = tuple((n for n in t.topology_sizes if n <= 32))
    configs = soundness_suite(sizes, seed=cfg.seed)
    rows, per_rule = comparison.score_composition_rules(
        configs, N=min(t.episodes, 4000), seed=cfg.seed
    )
    indep_violated = [r for r in rows if r["independence_violated"]]
    max_ratio = max(
        (r["independence_ratio"] for r in indep_violated if r["independence_ratio"] is not None),
        default=None,
    )
    metrics = {
        "n_configs": len(rows),
        "per_rule": per_rule,
        "independence_violated_configs": len(indep_violated),
        "independence_max_ratio": max_ratio,
        "tpag_hw_coverage": per_rule["tpag_hw"]["coverage"],
    }
    rec.write_csv("per_config.csv", rows)
    rec.write_json("per_rule.json", per_rule)
    return finalize(rec, cfg, RQ, "computed", metrics, provenance={"per_config": "per_config.csv"})
