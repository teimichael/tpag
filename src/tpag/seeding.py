"""Seeding for the TPAG replication package."""

from __future__ import annotations

import hashlib

import numpy as np


class SeedManager:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._generators: dict[str, np.random.Generator] = {}

    def _derive(self, name: str) -> int:
        h = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    def rng(self, name: str = "default") -> np.random.Generator:
        if name not in self._generators:
            self._generators[name] = np.random.default_rng(self._derive(name))
        return self._generators[name]

    def stream_seed(self, name: str) -> int:
        """A deterministic integer seed for an external sampler (e.g. an LLM
        backend) keyed by ``name``."""
        return self._derive(name) % (2**31 - 1)
