# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-submit audit of a Gemma 4 SWE agent submission package.

Every check mirrors a line of the project's pre-submit checklist. The audit is
deterministic and read-only, so an LLM agent can call it freely. Checks that
rest on an inferred harness schema are reported as warnings, not errors,
because ``HARNESS_README.md`` has not been confirmed.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import re
import struct
from typing import Any

import yaml

COMPETITION_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
HARNESS_TOOLS = (
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "submit_patch",
    "get_status",
)
# Tools a sub-agent may hold. run_command can mutate the repository, so it is
# excluded along with the editing tools.
READ_ONLY_TOOLS = ("read_file", "get_status")
MAIN_ADAPTER = "main_lora"
TOOL_ADAPTER = "tool_lora"
MAIN_LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
TOOL_LORA_MAX_RANK = 8
TARGET_MODULES = frozenset(
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
EXCLUDED_REPOS = ("encode__starlette",)
MAX_PACKAGE_BYTES = 3 * 2**30
MAX_TEMPERATURE = 0.3
MIN_MAX_TOKENS = 4096
MAX_ANALYZER_WORDS = 200
LAYER_KEY = re.compile(
    r"language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.lora_[AB]\.weight$"
)
REQUIRED_AGENT_FIELDS = ("name", "description", "model", "adapter", "instruction", "tools")


class _Include(str):
  """The target path of a ``!include`` tag."""


class _Loader(yaml.SafeLoader):
  pass


_Loader.add_constructor(
    "!include", lambda loader, node: _Include(loader.construct_scalar(node))
)


def _load_yaml(path: Path) -> Any:
  return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


def _finding(level: str, code: str, message: str) -> dict[str, str]:
  return {"level": level, "code": code, "message": message}


def read_safetensors_header(path: Path) -> dict[str, Any]:
  """Reads the JSON header of a safetensors file without loading tensors."""
  with open(path, "rb") as f:
    raw = f.read(8)
    if len(raw) < 8:
      raise ValueError("file shorter than the 8-byte header length")
    (length,) = struct.unpack("<Q", raw)
    if length > 100 * 2**20:
      raise ValueError(f"implausible header length {length}")
    return json.loads(f.read(length))


def _check_adapter(root: Path, name: str) -> list[dict[str, str]]:
  out: list[dict[str, str]] = []
  folder = root / "adapters" / name
  config_path = folder / "adapter_config.json"
  tensors_path = folder / "adapter_model.safetensors"
  for p in (config_path, tensors_path):
    if not p.is_file():
      out.append(_finding("error", "adapter_file_missing", f"missing {p.relative_to(root)}"))
  if out:
    return out
  config = json.loads(config_path.read_text(encoding="utf-8"))
  if "exclude_modules" in config:
    out.append(_finding("error", "exclude_modules", f"{name}: remove exclude_modules from adapter_config.json"))
  modules = set(config.get("target_modules") or [])
  if modules != TARGET_MODULES:
    out.append(_finding("error", "target_modules", f"{name}: target_modules {sorted(modules)} != the seven language-model projections"))
  base = str(config.get("base_model_name_or_path") or "")
  if base and "31b" not in base.lower():
    out.append(_finding("error", "wrong_base", f"{name}: trained on {base!r}, not a 31B base; shapes will not match"))
  rank = config.get("r")
  if name == MAIN_ADAPTER:
    for key, want in MAIN_LORA.items():
      if config.get(key) != want:
        out.append(_finding("error", "main_lora_param", f"{name}: {key}={config.get(key)!r}, expected {want!r}"))
  elif isinstance(rank, int) and rank > TOOL_LORA_MAX_RANK:
    out.append(_finding("warn", "tool_lora_rank", f"{name}: rank {rank} is above {TOOL_LORA_MAX_RANK}; a narrow adapter should stay low-rank"))
  try:
    header = read_safetensors_header(tensors_path)
  except (ValueError, OSError, json.JSONDecodeError) as e:
    return out + [_finding("error", "safetensors_header", f"{name}: {e}")]
  header.pop("__metadata__", None)
  if not header:
    out.append(_finding("error", "no_tensors", f"{name}: no tensors"))
  bad = [k for k in header if not LAYER_KEY.search(k)]
  if bad:
    out.append(_finding("error", "tensor_keys", f"{name}: unexpected tensor keys {bad[:3]}"))
  mismatched = [
      k
      for k, v in header.items()
      if k not in bad and v["shape"][0 if "lora_A" in k else 1] != rank
  ]
  if mismatched:
    out.append(_finding("error", "tensor_rank", f"{name}: tensors disagree with r={rank}: {mismatched[:3]}"))
  return out


def _check_agent_file(
    root: Path, path: Path, adapters: set[str], *, sub_agent: bool
) -> list[dict[str, str]]:
  out: list[dict[str, str]] = []
  rel = path.relative_to(root)
  try:
    spec = _load_yaml(path)
  except yaml.YAMLError as e:
    return [_finding("error", "yaml", f"{rel}: {e}")]
  if not isinstance(spec, dict):
    return [_finding("error", "yaml", f"{rel}: expected a mapping")]
  for field in REQUIRED_AGENT_FIELDS:
    if field not in spec:
      out.append(_finding("error", "agent_field", f"{rel}: missing field {field!r}"))
  if spec.get("model") != COMPETITION_MODEL:
    out.append(_finding("error", "model", f"{rel}: model {spec.get('model')!r} != {COMPETITION_MODEL!r}"))
  if spec.get("adapter") not in adapters:
    out.append(_finding("error", "adapter_name", f"{rel}: adapter {spec.get('adapter')!r} has no folder in adapters/ ({sorted(adapters)})"))
  allowed = READ_ONLY_TOOLS if sub_agent else HARNESS_TOOLS
  for tool in spec.get("tools") or []:
    if tool in allowed:
      continue
    if tool in HARNESS_TOOLS:
      out.append(_finding("error", "mutating_tool", f"{rel}: sub-agent holds {tool!r}; only {list(READ_ONLY_TOOLS)} are read-only"))
    else:
      near = difflib.get_close_matches(str(tool), HARNESS_TOOLS, n=1)
      hint = f" (did you mean {near[0]!r}?)" if near else ""
      out.append(_finding("error", "tool_name", f"{rel}: unknown tool {tool!r}{hint}"))
  instruction = spec.get("instruction")
  if isinstance(instruction, _Include):
    target = (path.parent / instruction).resolve()
    if not target.is_file():
      out.append(_finding("error", "include", f"{rel}: !include target {instruction!r} does not resolve"))
    elif sub_agent:
      words = len(target.read_text(encoding="utf-8").split())
      if words > MAX_ANALYZER_WORDS:
        out.append(_finding("warn", "analyzer_prompt_long", f"{rel}: prompt has {words} words; keep a role-specific prompt under {MAX_ANALYZER_WORDS}"))
  for sub in spec.get("subagents") or []:
    if not (root / "sub_agents" / f"{sub}.yaml").is_file():
      out.append(_finding("error", "subagent_name", f"{rel}: sub-agent {sub!r} has no sub_agents/{sub}.yaml"))
  return out


def audit_package(
    package_dir: Path, *, training_prompt: Path | None = None
) -> list[dict[str, str]]:
  """Audits a submission directory and returns a list of findings.

  Args:
    package_dir: The directory that will be zipped, holding ``agent.yaml``.
    training_prompt: The system prompt used in training. When given,
      ``prompts/system.md`` must match it byte for byte.

  Returns:
    Findings with ``level`` ``error`` or ``warn``. No error means the checks
    this module can run passed; it does not prove the harness accepts the
    package.
  """
  root = package_dir.resolve()
  out: list[dict[str, str]] = []
  if not (root / "agent.yaml").is_file():
    return [_finding("error", "layout", "agent.yaml is missing")]
  adapters_dir = root / "adapters"
  adapters = (
      {p.name for p in adapters_dir.iterdir() if p.is_dir()}
      if adapters_dir.is_dir()
      else set()
  )
  if MAIN_ADAPTER not in adapters:
    out.append(_finding("error", "layout", f"adapters/{MAIN_ADAPTER} is missing"))
  for name in sorted(adapters):
    out += _check_adapter(root, name)
  out += _check_agent_file(root, root / "agent.yaml", adapters, sub_agent=False)
  for sub in sorted((root / "sub_agents").glob("*.yaml")):
    out += _check_agent_file(root, sub, adapters, sub_agent=True)

  system = root / "prompts" / "system.md"
  if not system.is_file():
    out.append(_finding("error", "layout", "prompts/system.md is missing"))
  elif training_prompt is None:
    out.append(_finding("warn", "system_prompt_unverified", "no training prompt given; cannot confirm system.md matches training"))
  elif system.read_bytes() != training_prompt.read_bytes():
    out.append(_finding("error", "system_prompt", "prompts/system.md differs from the training system prompt"))

  sampling = root / "configs" / "sampling.yaml"
  if not sampling.is_file():
    out.append(_finding("warn", "layout", "configs/sampling.yaml is missing"))
  else:
    cfg = yaml.safe_load(sampling.read_text(encoding="utf-8")) or {}
    if cfg.get("temperature", 0) > MAX_TEMPERATURE:
      out.append(_finding("warn", "temperature", f"temperature {cfg.get('temperature')} is above {MAX_TEMPERATURE}"))
    if cfg.get("max_tokens", 0) < MIN_MAX_TOKENS:
      out.append(_finding("warn", "max_tokens", f"max_tokens {cfg.get('max_tokens')} is below {MIN_MAX_TOKENS}"))
  if not (root / "eval_config.yaml").is_file():
    out.append(_finding("warn", "layout", "eval_config.yaml is missing; it must also be tested locally"))

  total = 0
  for p in root.rglob("*"):
    if not p.is_file():
      continue
    size = p.stat().st_size
    total += size
    if size < 1_000_000 and p.suffix in (".yaml", ".md", ".json"):
      text = p.read_text(encoding="utf-8", errors="replace")
      for repo in EXCLUDED_REPOS:
        if repo in text:
          out.append(_finding("warn", "excluded_repo", f"{p.relative_to(root)} mentions excluded repo {repo}"))
  if total >= MAX_PACKAGE_BYTES:
    out.append(_finding("error", "size", f"package is {total / 2**30:.2f} GiB; the limit is 3 GiB"))
  return out


def passed(findings: list[dict[str, str]]) -> bool:
  return not any(f["level"] == "error" for f in findings)
