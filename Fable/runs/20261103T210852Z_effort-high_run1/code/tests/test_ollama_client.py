import json
import urllib.error

from fable_pipeline.config import OllamaConfig
from fable_pipeline.ollama_client import OllamaClient


class _Resp:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_unavailable_server_returns_none():
    def opener(req, timeout=None):
        raise urllib.error.URLError("refused")

    client = OllamaClient(OllamaConfig(), opener=opener)
    assert client.is_available() is False
    assert client.has_model("llava:7b") is False
    assert client.generate("hi") is None


def test_generate_uses_deterministic_options():
    seen = {}

    def opener(req, timeout=None):
        if req.full_url.endswith("/api/tags"):
            return _Resp({"models": [{"name": "qwen2.5:7b"}, {"name": "llava:7b"}]})
        seen.update(json.loads(req.data.decode()))
        return _Resp({"response": "  ok  "})

    cfg = OllamaConfig(seed=7, temperature=0.0, text_model="qwen2.5:7b")
    client = OllamaClient(cfg, opener=opener)
    assert client.is_available() and client.has_model("qwen2.5:7b") and not client.has_model("gemma")
    assert client.generate("prompt", images=[b"abc"]) == "ok"
    assert seen["options"] == {"temperature": 0.0, "seed": 7, "num_predict": cfg.num_predict}
    assert seen["stream"] is False and seen["images"] == ["YWJj"]
