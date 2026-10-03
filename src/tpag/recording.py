"""Recording for the TPAG replication package."""

from __future__ import annotations
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from . import __version__


@dataclass
class RunRecorder:
    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.setdefault("tpag_version", __version__)
        self.manifest.setdefault("python", sys.version.split()[0])

    def set(self, key: str, value: Any) -> None:
        self.manifest[key] = value

    def write_manifest(self) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str)
        )

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.run_dir / name
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def append_jsonl(self, name: str, row: dict[str, Any]) -> None:
        with (self.run_dir / name).open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def write_json(self, name: str, obj: Any) -> None:
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, default=str))

    def write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            (self.run_dir / name).write_text("")
            return
        keys = list({k for r in rows for k in r})
        with (self.run_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def record_trace(self, trace) -> None:
        self.write_jsonl("traces.jsonl", trace.to_jsonl())

    def record_certificate(self, cert) -> None:
        self.write_json("certificate.json", cert.to_dict())

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        self.write_json("metrics.json", metrics)
