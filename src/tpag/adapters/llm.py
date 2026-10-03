"""Llm for the TPAG replication package."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
_THINK_RE = re.compile("<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMResponse:
    content: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    thinking: str | None = None
    latency_s: float = 0.0


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` chain-of-thought blocks from a response."""
    return _THINK_RE.sub("", text).strip()


class LLMClient(Protocol):
    name: str

    def available(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]], **opts: Any) -> LLMResponse: ...


@dataclass
class OllamaClient:
    """Native Ollama ``/api/chat`` client (default backend).

    The default ``model`` is an approved-registry identifier; callers that run the
    evaluation always pass an explicit approved tag (see ``tpag.eval.config``)."""

    model: str = "qwen3.5:4b"
    base_url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    seed: int = 0
    think: bool = False
    timeout: float = 120.0
    keep_alive: str = "30m"

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def model_digest(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            for m in r.json().get("models", []):
                if m["name"] == self.model:
                    return m.get("digest", "")[:16]
        except Exception:
            pass
        return ""

    def chat(self, messages: list[dict[str, str]], **opts: Any) -> LLMResponse:
        import time

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": opts.get("think", self.think),
            "keep_alive": opts.get("keep_alive", self.keep_alive),
            "options": {
                "temperature": opts.get("temperature", self.temperature),
                "seed": opts.get("seed", self.seed),
            },
        }
        if opts.get("format") is not None:
            payload["format"] = opts["format"]
        t0 = time.time()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        latency = time.time() - t0
        msg = data.get("message", {})
        content = strip_thinking(msg.get("content", ""))
        return LLMResponse(
            content=content,
            model=self.model,
            raw=data,
            thinking=msg.get("thinking"),
            latency_s=latency,
        )


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[dict[str, Any]]:
    r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
    r.raise_for_status()
    return r.json().get("models", [])


def ollama_available(base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=5.0).status_code == 200
    except Exception:
        return False
