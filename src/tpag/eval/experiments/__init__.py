"""Init for the TPAG replication package."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...recording import RunRecorder
from ..config import ExpConfig
from ..schema import base_manifest, process_memory_mib, validate_metrics


class ResultsExist(RuntimeError):
    """Raised when a completed run directory exists and ``force`` was not passed."""


def make_recorder(
    cfg: ExpConfig,
    rq: str,
    subdir: str,
    *,
    argv: list[str] | None = None,
    force: bool = False,
    extra: dict | None = None,
) -> RunRecorder:
    """Create a recorder for one RQ.

    Refuses to overwrite a *completed* run (one with ``metrics.json``) unless
    ``force`` is set. Writes the manifest immediately with ``status='running'`` so
    that an interrupted run still leaves reproducible metadata next to whatever
    partial artifacts were streamed.
    """
    run_dir = cfg.out_path / subdir
    if (run_dir / "metrics.json").exists() and not force:
        raise ResultsExist(
            f"{run_dir}/metrics.json exists; pass --force to overwrite "
            f"(or choose a different --out)."
        )
    manifest = base_manifest(rq, cfg, argv=argv, extra=extra)
    manifest["status"] = "running"
    rec = RunRecorder(run_dir, manifest=manifest)
    rec._started = time.perf_counter()
    rec.write_manifest()  # metadata on disk before any work
    return rec


def finalize(
    rec: RunRecorder,
    cfg: ExpConfig,
    rq: str,
    status: str,
    metrics: dict[str, Any],
    provenance: dict | None = None,
) -> dict[str, Any]:
    """Assemble, validate, and persist the standard metrics record, stamping the
    run's wall-clock time and peak memory."""
    t0 = rec._started
    rec.set("status", status)
    rec.set("wall_seconds", round(time.perf_counter() - t0, 3))
    rec.set("peak_memory_mib", process_memory_mib())
    rec.set("metrics", metrics)
    rec.write_manifest()

    record: dict[str, Any] = {
        "rq": rq,
        "status": status,
        "tier": cfg.tier.name,
        "seed": cfg.seed,
        "config_hash": cfg.config_hash(),
        "benchmark_version": cfg.benchmark_version,
        "wall_seconds": rec.manifest["wall_seconds"],
        "metrics": metrics,
    }
    if provenance:
        record["provenance"] = provenance
    problems = validate_metrics(record)
    if problems:
        record["_schema_problems"] = problems  # surfaced, never silently dropped
    rec.record_metrics(record)  # writes metrics.json -> the "completed" marker
    return record


def jsonl_appender(rec: RunRecorder, name: str):
    """Return a function that appends one row to ``name`` (streaming intermediate
    artifacts so partial progress survives an interrupt). Truncates any stale file
    from a prior partial run on first use."""
    path = Path(rec.run_dir) / name
    if path.exists():
        path.unlink()

    def _append(row: dict) -> None:
        rec.append_jsonl(name, row)

    return _append


__all__ = ["make_recorder", "finalize", "jsonl_appender", "ResultsExist"]
