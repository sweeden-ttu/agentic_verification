import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _post(url, headers, payload, timeout=600, retries=5):
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last = RuntimeError(f"HTTP {e.code}: {body[:500]}")
            if e.code in (400, 401, 403, 404):
                raise last
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(60, 2 ** attempt))
    raise last


class OpenAICompat:
    def __init__(self, spec):
        self.spec = spec
        self.base_url = spec["base_url"].rstrip("/")
        self.model = spec.get("model", "default")
        key = os.environ.get(spec["api_key_env"]) if spec.get("api_key_env") else None
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        self.supports_logprobs = bool(spec.get("logprobs"))

    def chat(self, prompt, max_tokens, temperature, logprobs=False, seed=None):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 20
        payload.update(self.spec.get("extra_body", {}))
        d = _post(self.base_url + "/chat/completions", self.headers, payload)
        ch = d["choices"][0]
        msg = ch.get("message", {})
        text = msg.get("content") or ""
        lp = None
        if logprobs and ch.get("logprobs") and ch["logprobs"].get("content"):
            lp = [
                {"token": t.get("token", ""), "top": [(a.get("token", ""), a.get("logprob", -99.0)) for a in t.get("top_logprobs", [])]}
                for t in ch["logprobs"]["content"]
            ]
        return {"text": text, "logprobs": lp, "finish": ch.get("finish_reason"), "usage": d.get("usage", {})}


class Anthropic:
    def __init__(self, spec):
        self.spec = spec
        self.model = spec["model"]
        self.url = spec.get("base_url", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        key = os.environ.get(spec.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not key:
            raise RuntimeError(f"{spec['id']}: set {spec.get('api_key_env', 'ANTHROPIC_API_KEY')}")
        self.headers = {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}
        self.supports_logprobs = False
        self.send_temperature = True

    def chat(self, prompt, max_tokens, temperature, logprobs=False, seed=None):
        payload = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if self.send_temperature:
            payload["temperature"] = temperature
        try:
            d = _post(self.url, self.headers, payload)
        except RuntimeError as e:
            if self.send_temperature and "temperature" in str(e):
                self.send_temperature = False
                return self.chat(prompt, max_tokens, temperature, logprobs, seed)
            raise
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        return {"text": text, "logprobs": None, "finish": d.get("stop_reason"), "usage": d.get("usage", {})}


def make(spec):
    if spec["provider"] == "anthropic":
        return Anthropic(spec)
    if spec["provider"] == "openai":
        return OpenAICompat(spec)
    raise ValueError(f"unknown provider {spec['provider']}")


class Session:
    def __init__(self, spec):
        self.spec = spec
        self.proc = None

    def __enter__(self):
        launch = self.spec.get("launch")
        if launch:
            cmd = [str(ROOT / c) if (c.startswith(".tools/") or c.startswith("models/")) else c for c in launch["cmd"]]
            log = open(ROOT / "runs" / f"server_{self.spec['id']}.log", "w")
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            deadline = time.time() + launch.get("timeout_s", 600)
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.spec['id']}: server exited, see runs/server_{self.spec['id']}.log")
                try:
                    with urllib.request.urlopen(launch["health"], timeout=5) as r:
                        if r.status == 200 and b"ok" in r.read():
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError(f"{self.spec['id']}: server not healthy in time")
        return make(self.spec)

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False


def load_models(path, only=None):
    cfg = json.loads(Path(path).read_text())
    specs = [m for m in cfg["models"] if m.get("enabled", True)]
    if only:
        keep = set(only)
        specs = [m for m in specs if m["id"] in keep]
    return sorted(specs, key=lambda m: -m["rank"])
