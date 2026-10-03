"""Model zoo for the TPAG replication package."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from ..adapters.llm import list_ollama_models, ollama_available
from .config import TIER_RANK, TIERS, ExpConfig, ModelSpec

_PARAM_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([BM])", re.IGNORECASE)


def parse_params_b(detail: str | None) -> float | None:
    """Parse a parameter-size string (e.g. ``'14.8B'``, ``'3.2B'``, ``'770M'``)
    into a billions float, or ``None`` if unparseable."""
    if not detail:
        return None
    m = _PARAM_RE.search(detail)
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000.0 if m.group(2).upper() == "M" else val


@dataclass
class ZooStatus:
    ollama_up: bool
    # ``available`` = present approved models permitted by the hard size cap.
    available: list[ModelSpec] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # approved tags absent
    excluded_oversized: list[ModelSpec] = field(default_factory=list)  # present but >= cap
    extra: list[str] = field(default_factory=list)  # present, not approved
    reason: str = ""

    def eligible(self, max_params_b: float = 70.0) -> list[ModelSpec]:
        return [m for m in self.available if m.params_b < max_params_b]

    def by_tier(self, cfg: ExpConfig) -> dict[str, list[ModelSpec]]:
        """Group available models by their *explicit* (assigned) scale tier."""
        out: dict[str, list[ModelSpec]] = {t: [] for t in TIERS}
        for m in self.available:
            out.setdefault(m.resolved_tier(cfg), []).append(m)
        return out

    def by_family(self) -> dict[str, list[ModelSpec]]:
        out: dict[str, list[ModelSpec]] = {}
        for m in self.available:
            out.setdefault(m.family, []).append(m)
        return out


def probe_zoo(cfg: ExpConfig) -> ZooStatus:
    """Detect which approved models are present in the local Ollama. Enforces the
    hard size cap (>= ``cfg.max_params_b`` => excluded). Fail-safe: if Ollama is
    unreachable, returns ``ollama_up=False`` with everything missing."""
    if not ollama_available(cfg.ollama_url):
        return ZooStatus(
            ollama_up=False,
            missing=[m.tag for m in cfg.zoo],
            reason=f"Ollama not reachable at {cfg.ollama_url}",
        )
    try:
        present = list_ollama_models(cfg.ollama_url)
    except Exception as exc:  # pragma: no cover - network dependent
        return ZooStatus(
            ollama_up=False, missing=[m.tag for m in cfg.zoo], reason=f"{type(exc).__name__}: {exc}"
        )

    present_by_tag = {m["name"]: m for m in present}
    approved_tags = {m.tag for m in cfg.zoo}
    available: list[ModelSpec] = []
    excluded: list[ModelSpec] = []
    missing: list[str] = []
    for spec in cfg.zoo:
        info = present_by_tag.get(spec.tag)
        if info is None:
            missing.append(spec.tag)
            continue
        detected = parse_params_b(info.get("details", {}).get("parameter_size"))
        resolved = ModelSpec(
            spec.tag,
            spec.family,
            detected if detected else spec.params_b,
            note=spec.note,
            tier=spec.tier,
        )  # preserve the assigned tier
        if resolved.params_b >= cfg.max_params_b:
            excluded.append(resolved)  # hard cap: never used in experiments
        else:
            available.append(resolved)
    extra = sorted(set(present_by_tag) - approved_tags)
    return ZooStatus(
        ollama_up=True,
        available=available,
        missing=missing,
        excluded_oversized=excluded,
        extra=extra,
    )


# --------------------------------------------------------------------------
# Required-model validation (preflight).
# --------------------------------------------------------------------------
def required_tags(cfg: ExpConfig) -> list[str]:
    """The approved Ollama identifiers the experiments require."""
    return [m.tag for m in cfg.zoo]


def missing_required(cfg: ExpConfig, status: ZooStatus) -> list[str]:
    """Approved tags not currently available locally."""
    have = {m.tag for m in status.available}
    return [t for t in required_tags(cfg) if t not in have]


# --------------------------------------------------------------------------
# RQ-E3 condition matrix (within each scale tier).
# --------------------------------------------------------------------------
def _condition(name: str, diversity: str, k: int, specs: list[ModelSpec], tier: str) -> dict:
    params = [s.params_b for s in specs]
    return {
        "name": name,
        "diversity": diversity,
        "k": k,
        "models": [s.tag for s in specs],
        "families": [s.family for s in specs],
        "params_b": params,
        "mean_params_b": sum(params) / len(params),
        "max_params_b": max(params),
        "size_tier": tier,  # the assigned scale tier of this condition
    }


def build_conditions(
    cfg: ExpConfig,
    status: ZooStatus,
    *,
    ks: tuple[int, ...] | None = None,
    diversity_modes: tuple[str, ...] | None = None,
    tiers: tuple[str, ...] | None = None,
    max_per_tier: int | None = None,
) -> list[dict]:
    """Enumerate RQ-E3 model-level conditions *within each scale tier*.

    For each tier (small/mid/large) and each width ``k``:
      * ``same_model``   -- ``k`` copies of one approved model (shared blind spot);
      * ``cross_family`` -- ``k`` models from distinct approved families at that tier
        (design diversity).
    Conditions are ordered small-first (tier rank, then params), so the
    ``max_conditions`` cap favors cheaper scales -- larger scales are the
    capability/robustness check. This yields a clean small/mid/large x family matrix.

    The keyword args let a single *block* of a fractional design request its own
    scales/widths/diversity and cap conditions per tier (``max_per_tier``, applied
    small-first per tier); they default to the ``cfg`` values (back-compat).
    """
    ks = ks if ks is not None else cfg.tier.ks
    diversity_modes = diversity_modes if diversity_modes is not None else cfg.diversity_modes
    want_tiers = tiers if tiers is not None else TIERS

    avail = status.eligible(cfg.max_params_b)
    if not avail:
        return []
    # group the *eligible* (< hard cap) models by their assigned tier
    by_tier: dict[str, list[ModelSpec]] = {t: [] for t in TIERS}
    for m in avail:
        by_tier.setdefault(m.resolved_tier(cfg), []).append(m)
    conditions: list[dict] = []

    for tier in want_tiers:
        tier_models = sorted(by_tier.get(tier, []), key=lambda s: (s.params_b, s.tag))
        if not tier_models:
            continue
        tier_conditions: list[dict] = []
        for k in ks:
            if "same_model" in diversity_modes:
                for spec in tier_models:
                    tier_conditions.append(
                        _condition(f"same[{spec.tag}]x{k}", "same_model", k, [spec] * k, tier)
                    )
            if "cross_family" in diversity_modes:
                # one representative per family at this tier
                reps: dict[str, ModelSpec] = {}
                for spec in tier_models:
                    reps.setdefault(spec.family, spec)
                rep_list = list(reps.values())
                for combo in combinations(rep_list, min(k, len(rep_list))):
                    if len(combo) < k:
                        continue
                    tier_conditions.append(
                        _condition(
                            "cross[" + "+".join(s.tag for s in combo) + "]",
                            "cross_family",
                            k,
                            list(combo),
                            tier,
                        )
                    )
        tier_conditions.sort(key=lambda c: (c["mean_params_b"], c["max_params_b"]))
        if max_per_tier is not None:
            tier_conditions = tier_conditions[:max_per_tier]
        conditions.extend(tier_conditions)

    conditions.sort(key=lambda c: (TIER_RANK.get(c["size_tier"], 9), c["mean_params_b"]))
    return conditions


# --------------------------------------------------------------------------
# Fractional design: expand experiment *blocks* into a single group list.
# --------------------------------------------------------------------------
def _block_axes(block: dict, cfg: ExpConfig) -> dict:
    """Resolve a block's axes, each defaulting to the cfg/tier value."""
    return {
        "name": block.get("name", "block"),
        "tiers": tuple(block.get("tiers", TIERS)),
        "diversity": tuple(block.get("diversity", cfg.diversity_modes)),
        "ks": tuple(block.get("ks", cfg.tier.ks)),
        "domains": tuple(block.get("domains", cfg.domains)),
        "hardenings": tuple(block.get("hardenings", cfg.hardenings)),
        "strategies": tuple(block.get("strategies", cfg.injection_strategies)),
        "n_requests": int(block.get("n_requests", cfg.tier.n_requests)),
        "n_seeds": int(block.get("n_seeds", cfg.tier.n_seeds)),
        "injection_fractions": tuple(
            block.get("injection_fractions", cfg.tier.injection_fractions)
        ),
        "max_per_tier": block.get("max_conditions_per_tier"),
    }


def expand_blocks(cfg: ExpConfig, status: ZooStatus) -> list[dict]:
    """Expand ``cfg.blocks`` into a de-duplicated, swap-minimizing group list.

    Each group carries its block's ``n_requests`` / ``n_seeds`` /
    ``injection_fractions`` (so a block can be powered while another probes an axis
    cheaply). Operating conditions identical in ``(models, domain, hardening,
    strategy)`` are de-duplicated across blocks (first block wins). Groups are ordered
    by ``(block_index, tier rank, model-set)`` so a model stays resident across a
    group's fractions/seeds and across adjacent groups (works with ``keep_alive``)."""
    groups: list[dict] = []
    seen: set[tuple] = set()
    for bi, block in enumerate(cfg.blocks):
        ax = _block_axes(block, cfg)
        conds = build_conditions(
            cfg,
            status,
            ks=ax["ks"],
            diversity_modes=ax["diversity"],
            tiers=ax["tiers"],
            max_per_tier=ax["max_per_tier"],
        )
        for c in conds:
            for d in ax["domains"]:
                for h in ax["hardenings"]:
                    for s in ax["strategies"]:
                        key = (tuple(c["models"]), d, h, s)
                        if key in seen:
                            continue
                        seen.add(key)
                        groups.append(
                            {
                                "condition": c,
                                "domain": d,
                                "hardening": h,
                                "strategy": s,
                                "block": ax["name"],
                                "block_index": bi,
                                "n_requests": ax["n_requests"],
                                "n_seeds": ax["n_seeds"],
                                "injection_fractions": list(ax["injection_fractions"]),
                            }
                        )
    groups.sort(
        key=lambda g: (
            g["block_index"],
            TIER_RANK.get(g["condition"]["size_tier"], 9),
            tuple(sorted(g["condition"]["models"])),
        )
    )
    return groups


def estimate_block_calls(cfg: ExpConfig, status: ZooStatus) -> dict:
    """LLM-call estimate for a block (fractional) design. Contacts no model."""
    groups = expand_blocks(cfg, status)
    calls = 0
    per_block: dict[str, int] = {}
    for g in groups:
        k = len(g["condition"]["models"])  # one call per guard per request
        c = len(g["injection_fractions"]) * g["n_seeds"] * g["n_requests"] * k
        calls += c
        per_block[g["block"]] = per_block.get(g["block"], 0) + c
    return {
        "mode": "blocks",
        "n_blocks": len(cfg.blocks),
        "n_groups": len(groups),
        "approx_llm_calls": calls,
        "calls_per_block": per_block,
        # rough band with keep_alive warm models (small ~0.5s, large ~4s)
        "approx_wall_hours_low": round(calls * 0.5 / 3600.0, 2),
        "approx_wall_hours_high": round(calls * 4.0 / 3600.0, 2),
        "note": "estimate only; no model contacted. Large-tier reasoning models dominate.",
    }


def pull_hints(status: ZooStatus) -> list[str]:
    """``ollama pull`` commands for missing approved models (manual; never auto-run)."""
    return [f"ollama pull {tag}" for tag in status.missing]


__all__ = [
    "parse_params_b",
    "ZooStatus",
    "probe_zoo",
    "required_tags",
    "missing_required",
    "build_conditions",
    "expand_blocks",
    "estimate_block_calls",
    "pull_hints",
]
