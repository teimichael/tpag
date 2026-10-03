"""Run or inspect the experiments retained in the replication package."""

from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path

DRIVERS = {
    "e1": "rq_e1_calibration",
    "e2": "rq_e2_topology",
    "e3": "rq_e3_llm_correlation",
    "e4": "rq_e4_baselines",
    "e5": "rq_e5_overhead_integration",
    "e6": "rq_e6_estimation_ablation",
    "e9": "rq_e9_hard_containment",
}


def cmd_models(args: argparse.Namespace) -> int:
    from .eval import model_zoo
    from .eval.config import load_config

    cfg = load_config(path=args.config, tier=args.tier)
    status = model_zoo.probe_zoo(cfg)
    if not status.ollama_up:
        print(f"Ollama NOT reachable: {status.reason}")
        print("Candidate models to pull for the RQ-E3 zoo:")
        for h in model_zoo.pull_hints(status):
            print(f"  {h}")
        return 1
    bt = status.by_tier(cfg)
    print(
        f"Ollama up at {cfg.ollama_url}. {len(status.available)}/{len(cfg.zoo)} approved model(s) present: small={len(bt['small'])}, mid={len(bt['mid'])}, large={len(bt['large'])}."
    )
    for m in sorted(status.available, key=lambda s: (s.resolved_tier(cfg), s.params_b)):
        print(f"  - {m.tag:<22} {m.params_b:>5.1f}B  [{m.family}] {m.resolved_tier(cfg)}")
    if status.excluded_oversized:
        print(f"Excluded by hard cap (>= {cfg.max_params_b}B; NOT used):")
        for m in status.excluded_oversized:
            print(f"  - {m.tag:<22} {m.params_b:>5.1f}B  [{m.family}]")
    if status.missing:
        print("Missing approved models (run `ollama pull`; never auto-downloaded):")
        for h in model_zoo.pull_hints(status):
            print(f"  {h}")
    conds = model_zoo.build_conditions(cfg, status)
    print(f"\n{len(conds)} model-level RQ-E3 condition(s) buildable (small-first).")
    print("Tip: `tpag preflight` fails with a clear error if any approved model is missing.")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Validate the approved registry + Ollama availability. Exits non-zero with a
    clear error if anything required is missing (for gating runs / CI)."""
    from .eval import model_zoo
    from .eval.config import load_config, validate_registry

    cfg = load_config(path=args.config, tier=args.tier)
    try:
        validate_registry(cfg.zoo)
    except ValueError as exc:
        print(f"PREFLIGHT FAIL: approved-model registry is invalid: {exc}")
        return 2
    status = model_zoo.probe_zoo(cfg)
    if not status.ollama_up:
        print(f"PREFLIGHT FAIL: {status.reason}")
        print("Start Ollama (`ollama serve`) and pull the approved models:")
        for h in model_zoo.pull_hints(status):
            print(f"  {h}")
        return 2
    missing = model_zoo.missing_required(cfg, status)
    if missing:
        print(f"PREFLIGHT FAIL: {len(missing)}/{len(cfg.zoo)} approved model(s) not installed.")
        print("Pull them (manual; nothing is auto-downloaded):")
        for tag in missing:
            print(f"  ollama pull {tag}")
        return 3
    if status.excluded_oversized:
        print(
            "WARNING: oversized models present and will be ignored: "
            + ", ".join((m.tag for m in status.excluded_oversized))
        )
    n_fam = len({m.family for m in cfg.zoo})
    n_scale = len({m.resolved_tier(cfg) for m in cfg.zoo})
    print(
        f"PREFLIGHT OK: all {len(cfg.zoo)} approved models present ({n_fam} families x {n_scale} scales)."
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("models", "preflight"):
        q = sub.add_parser(name)
        q.add_argument("--tier", choices=("quick", "full"), default="full")
        q.add_argument("--config")
        q.set_defaults(func=cmd_models if name == "models" else cmd_preflight)
    q = sub.add_parser("run-ext", help="run a retained experiment; writes only to outputs/")
    q.add_argument("rq", choices=tuple(DRIVERS))
    q.add_argument("--tier", choices=("quick", "full"), default="full")
    q.add_argument("--seed", type=int)
    q.add_argument("--config")
    q.add_argument("--out", default="outputs/run")
    q.add_argument("--no-llm", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command != "run-ext":
        return args.func(args)
    from .eval.config import REPO_ROOT, load_config

    out = Path(args.out)
    target = out.resolve()
    try:
        target.relative_to((REPO_ROOT / "outputs").resolve())
    except ValueError:
        p.error("--out must be inside this package's outputs/ directory")
    cfg = load_config(path=args.config, tier=args.tier, seed=args.seed, out=args.out)
    if args.dry_run:
        if args.rq == "e3":
            from .eval.experiments.rq_e3_llm_correlation import recorded_groups

            groups = recorded_groups(cfg)
            estimate = {
                "n_groups": len(groups),
                "n_points": sum(len(g["injection_fractions"]) for g in groups),
                "llm_calls": sum(
                    len(g["injection_fractions"])
                    * g["n_requests"]
                    * g["n_seeds"]
                    * g["condition"]["k"]
                    for g in groups
                ),
                "design": "reported operating points; no model contacted",
            }
        else:
            estimate = {
                "experiment": args.rq,
                "tier": cfg.tier.name,
                "seed": cfg.seed,
                "output": args.out,
            }
        print(json.dumps(estimate, indent=2))
        return 0
    driver = importlib.import_module(".eval.experiments." + DRIVERS[args.rq], "tpag")
    result = driver.run(
        cfg, use_llm=not args.no_llm, argv=["tpag", "run-ext", args.rq], force=args.force
    )
    print(json.dumps(result, indent=2))
    if args.rq == "e3" and result["status"] != "measured":
        return 1
    return 0
