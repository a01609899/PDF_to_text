"""Minimal, dependency-free client for a local Ollama server.

Only ``/api/tags`` (availability check) and ``/api/generate`` are used. All
calls are made with ``temperature`` and ``seed`` from the config so results are
deterministic for the same model/prompt/image. Any failure returns ``None``
and is logged; callers must treat LLM output as optional.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .config import OllamaConfig

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, opener: Optional[Callable[..., Any]] = None):
        self.cfg = cfg
        self._opener = opener or urllib.request.urlopen
        self._available: Optional[bool] = None
        self._models: List[str] = []

    # ------------------------------------------------------------------ http
    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        url = self.cfg.url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with self._opener(req, timeout=timeout or self.cfg.timeout) as resp:
            body = resp.read()
        return json.loads(body.decode("utf-8")) if body else {}

    # ------------------------------------------------------------------- api
    def is_available(self) -> bool:
        if self._available is None:
            try:
                resp = self._request("/api/tags", timeout=5.0)
                self._models = sorted(m.get("name", "") for m in resp.get("models", []))
                self._available = True
            except Exception as exc:  # noqa: BLE001
                log.warning("Ollama not reachable at %s (%s); LLM steps are skipped", self.cfg.url, exc)
                self._available = False
        return bool(self._available)

    def has_model(self, name: str) -> bool:
        if not self.is_available():
            return False
        base = name.split(":")[0]
        return any(m == name or m.split(":")[0] == base for m in self._models) or not self._models

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        images: Optional[List[bytes]] = None,
        num_predict: Optional[int] = None,
        system: Optional[str] = None,
    ) -> Optional[str]:
        model = model or self.cfg.text_model
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                "num_predict": num_predict or self.cfg.num_predict,
            },
        }
        if images:
            payload["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        if system:
            payload["system"] = system
        try:
            resp = self._request("/api/generate", payload)
        except urllib.error.URLError as exc:
            log.warning("Ollama request failed (%s)", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama request failed (%s)", exc)
            return None
        text = resp.get("response")
        return text.strip() if isinstance(text, str) else None
