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

"""Tools for the LoRA improvement loop.

Every tool returns a JSON string with ``{"status": "ok", ...}`` or ``{"status":
"error", "error_type": ..., "error_message": ...}``, like the SWE harness tools
in the sibling ``swe-patch-team`` sample. All paths are relative to
``LOOP_WORKDIR`` (default ``./work``) and traversal is rejected.

The tools never run training and never submit. Training is done by the
baseline notebook, which owns the ``TrainEpoch`` and ``Evaluate`` steps of the
spec. These tools verify what the notebook did, plan the next job, and render
a candidate submission. ``Submit`` and ``Judge`` are deliberately not tools.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from . import package_checks
from . import spec

MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10_000
ADOPTED_STATE = "adopted_spec_state.json"
CANDIDATE_DIR = "candidate_submission"

# Fixed by the project: the agent may not change these.
LOCKED = {
    "base_model": package_checks.COMPETITION_MODEL,
    "lora": dict(package_checks.MAIN_LORA),
    "target_modules": sorted(package_checks.TARGET_MODULES),
    "grad_accum": 8,
    "max_len": 5120,
    "loss": "model tokens only",
    "scope": "language model only",
}
# Ranges the agent may move within.
LR_RANGE = {"main_lora": (5e-5, 2e-4), "tool_lora": (1e-5, 1e-4)}
N_TRAIN_RANGE = (8, 5000)
N_VAL_RANGE = (8, 200)


def workdir() -> Path:
  return Path(os.environ.get("LOOP_WORKDIR", "work")).resolve()


def _ok(**fields: Any) -> str:
  return json.dumps({"status": "ok", **fields})


def _error(error_type: str, message: str, **details: Any) -> str:
  payload: dict[str, Any] = {
      "status": "error",
      "error_type": error_type,
      "error_message": message,
  }
  if details:
    payload["details"] = details
  return json.dumps(payload)


class _PathError(ValueError):
  pass


def _resolve(relpath: str) -> Path:
  """Maps a model-supplied path into the working directory."""
  if not relpath or "\x00" in relpath:
    raise _PathError("path must be non-empty.")
  if ".." in Path(relpath).parts or Path(relpath).is_absolute():
    raise _PathError("Absolute paths and '..' are not allowed.")
  root = workdir()
  resolved = (root / relpath).resolve()
  if resolved != root and root not in resolved.parents:
    raise _PathError("Path resolves outside the working directory.")
  return resolved


def today() -> datetime.date:
  """Returns the calendar date, or ``LOOP_TODAY`` (ISO format) when set."""
  override = os.environ.get("LOOP_TODAY")
  return datetime.date.fromisoformat(override) if override else datetime.date.today()


def _load_monitor() -> tuple[spec.SpecMonitor, bool]:
  """Loads the adopted behavior in memory, caught up to today's date.

  The result is never saved, so a status check cannot add ``Tick`` steps to
  the adopted trace that the notebook has not seen.
  """
  path = workdir() / ADOPTED_STATE
  adopted = path.is_file()
  if adopted:
    state = json.loads(path.read_text(encoding="utf-8"))
    mon = spec.SpecMonitor(state=state)
  else:
    mon = spec.SpecMonitor(prophecy="lose")
  mon.catch_up(today())
  return mon, adopted


def _count(mon: spec.SpecMonitor, action: str) -> int:
  return sum(t["action"] == action for t in mon.s["trace"])


def spec_status() -> str:
  """Reports the loop state and which spec actions are enabled today.

  The prophecy variable is not shown: it is a ghost value, and keeping it out
  of the model's view keeps it from influencing decisions.

  Returns:
    A JSON string with the state, enabled actions, counts and recent actions.
  """
  mon, adopted = _load_monitor()
  return _ok(
      adopted_notebook_run=adopted,
      state={k: v for k, v in mon.view().items() if k != "p"},
      enabled={
          a: mon.enabled(a)
          for a in ("TrainEpoch", "Evaluate", "Package", "Submit", "Tick")
      },
      epochs_trained=_count(mon, "TrainEpoch"),
      jobs_planned=len(list((workdir() / "jobs").glob("job_*.json"))),
      recent_actions=[t["action"] for t in mon.s["trace"][-8:]],
  )


def list_artifacts() -> str:
  """Lists files in the working directory (read-only).

  Returns:
    A JSON string with up to 150 relative paths.
  """
  root = workdir()
  if not root.is_dir():
    return _ok(paths=[])
  paths = sorted(
      str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
  )
  return _ok(paths=paths[:150], truncated=len(paths) > 150)


def read_artifact(
    filepath: str, start_line: int | None = None, end_line: int | None = None
) -> str:
  """Reads part of a file in the working directory (1-indexed, inclusive).

  Args:
    filepath: Path relative to the working directory.
    start_line: First line to return. Defaults to 1.
    end_line: Last line to return. Defaults to the end of the file.

  Returns:
    A JSON string with content and line bounds; at most 150 lines.
  """
  try:
    path = _resolve(filepath)
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not path.is_file():
    return _error("FileNotFound", f"No such file: {filepath}")
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  first = max(1, start_line or 1)
  last = min(len(lines), end_line or len(lines))
  chosen = lines[first - 1 : min(last, first - 1 + MAX_FILE_LINES)]
  content = "\n".join(chosen)[:MAX_FILE_CHARS]
  return _ok(
      filepath=filepath,
      content=content,
      start_line=first,
      end_line=first + len(chosen) - 1,
      total_lines=len(lines),
  )


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """Compares trained epochs with the epoch-0 baseline row."""
  baseline = next((r for r in rows if r.get("epoch") == 0), None)
  trained = [r for r in rows if r.get("epoch", 0) > 0]
  if baseline is None or not trained:
    return {"improved_over_baseline": None, "reason": "need epoch 0 and one trained epoch"}
  best_acc = max(r["tool_acc"] for r in trained)
  improved = (
      trained[-1]["val_loss"] < baseline["val_loss"]
      and best_acc >= baseline["tool_acc"]
  )
  return {
      "improved_over_baseline": improved,
      "baseline": baseline,
      "last": trained[-1],
      "best_tool_acc": best_acc,
      "epochs": len(trained),
  }


def import_notebook_run(spec_state_path: str, metrics_path: str = "metrics.json") -> str:
  """Verifies a notebook run against the spec and adopts it as the loop state.

  The notebook's ``spec_state.json`` trace is replayed step by step. A trace
  with any step the spec does not allow is rejected, and so is one that does
  not extend the previously adopted trace.

  Args:
    spec_state_path: The notebook's ``spec_state.json``, relative to the
      working directory.
    metrics_path: The notebook's ``metrics.json``, relative to the working
      directory.

  Returns:
    A JSON string with the verification result and a metrics summary.
  """
  try:
    state_file, metrics_file = _resolve(spec_state_path), _resolve(metrics_path)
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not state_file.is_file():
    return _error("FileNotFound", f"No such file: {spec_state_path}")
  try:
    new = json.loads(state_file.read_text(encoding="utf-8"))
    trace = new["trace"]
  except (ValueError, KeyError) as e:
    return _error("ValidationError", f"not a spec_state.json: {e}")
  problems = spec.check_trace(trace)
  if problems:
    return _error("TraceViolation", "trace is not in the spec language", problems=problems[:10])
  adopted = workdir() / ADOPTED_STATE
  if adopted.is_file():
    old = json.loads(adopted.read_text(encoding="utf-8"))["trace"]
    if not spec.trace_extends(old, trace):
      return _error("TraceDiverged", "the new trace does not extend the adopted trace")
  shutil.copyfile(state_file, adopted)
  rows = []
  if metrics_file.is_file():
    rows = json.loads(metrics_file.read_text(encoding="utf-8"))
  return _ok(
      verified_steps=len(trace),
      state={k: v for k, v in spec.view(new).items() if k != "p"},
      metrics=summarize_metrics(rows),
  )


def validate_job_config(cfg: dict[str, Any]) -> list[str]:
  """Returns the reasons a proposed job leaves the allowed ranges."""
  problems: list[str] = []
  adapter = cfg.get("adapter")
  if adapter not in LR_RANGE:
    return [f"adapter must be one of {sorted(LR_RANGE)}"]
  lo, hi = LR_RANGE[adapter]
  if not lo <= cfg["learning_rate"] <= hi:
    problems.append(f"learning_rate for {adapter} must be in [{lo}, {hi}]")
  if not N_TRAIN_RANGE[0] <= cfg["n_train"] <= N_TRAIN_RANGE[1]:
    problems.append(f"n_train must be in {N_TRAIN_RANGE}")
  if not N_VAL_RANGE[0] <= cfg["n_val"] <= N_VAL_RANGE[1]:
    problems.append(f"n_val must be in {N_VAL_RANGE}")
  if not cfg["rationale"].strip():
    problems.append("rationale must not be empty")
  for repo in cfg["extra_exclude_repos"]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", repo):
      problems.append(f"exclude repo {repo!r} must look like owner__name")
  return problems


def plan_next_job(
    adapter: str,
    learning_rate: float,
    n_train: int,
    n_val: int,
    rationale: str,
    extra_exclude_repos: list[str] | None = None,
) -> str:
  """Writes the next training job for the notebook to run.

  The job is refused when the spec does not enable ``TrainEpoch`` today (the
  weekly budget is spent, the deadline passed, or a package is waiting), when
  an earlier job has not run yet, or when a value leaves its allowed range.
  Planning does not change the spec state; the notebook's ``TrainEpoch`` does.

  Args:
    adapter: ``main_lora`` or ``tool_lora``.
    learning_rate: Learning rate. ``main_lora`` allows 5e-5 to 2e-4;
      ``tool_lora`` allows 1e-5 to 1e-4.
    n_train: Number of training trajectories.
    n_val: Number of validation trajectories.
    rationale: Why this change should improve the validation score.
    extra_exclude_repos: Further repos, as ``owner__name``, to drop from the
      data on top of ``encode__starlette``.

  Returns:
    A JSON string with the job path and the environment the notebook reads.
  """
  cfg = dict(
      adapter=adapter,
      learning_rate=learning_rate,
      n_train=n_train,
      n_val=n_val,
      rationale=rationale,
      extra_exclude_repos=list(extra_exclude_repos or []),
  )
  mon, _ = _load_monitor()
  if not mon.enabled("TrainEpoch"):
    return _error(
        "SpecViolation",
        "TrainEpoch is not enabled today; wait for a budget refill or a Tick.",
        state={k: v for k, v in mon.view().items() if k != "p"},
    )
  jobs = workdir() / "jobs"
  existing = sorted(jobs.glob("job_*.json"))
  if len(existing) > _count(mon, "TrainEpoch"):
    return _error("JobPending", f"{existing[-1].name} has not been run by the notebook yet.")
  problems = validate_job_config(cfg)
  if problems:
    return _error("ValidationError", "job leaves the allowed ranges", problems=problems)
  jobs.mkdir(parents=True, exist_ok=True)
  path = jobs / f"job_{len(existing) + 1:03d}.json"
  job = {
      "config": cfg,
      "locked": LOCKED,
      "env": {
          "LR": repr(learning_rate),
          "N_TRAIN": str(n_train),
          "N_VAL": str(n_val),
          "MAX_LEN": str(LOCKED["max_len"]),
          "EXTRA_EXCLUDE_REPOS": ",".join(cfg["extra_exclude_repos"]),
      },
      "spec_state": {k: v for k, v in mon.view().items() if k != "p"},
  }
  path.write_text(json.dumps(job, indent=1), encoding="utf-8")
  return _ok(job_path=str(path.relative_to(workdir())), env=job["env"])


def audit_package(package_dir: str, training_prompt_path: str = "") -> str:
  """Runs the pre-submit checks on a package directory (read-only).

  Args:
    package_dir: The package directory, relative to the working directory.
    training_prompt_path: The system prompt used in training, relative to the
      working directory. When given, ``prompts/system.md`` must match it.

  Returns:
    A JSON string with ``passed`` and a list of findings.
  """
  try:
    root = _resolve(package_dir)
    prompt = _resolve(training_prompt_path) if training_prompt_path else None
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not root.is_dir():
    return _error("FileNotFound", f"No such directory: {package_dir}")
  if prompt is not None and not prompt.is_file():
    return _error("FileNotFound", f"No such file: {training_prompt_path}")
  findings = package_checks.audit_package(root, training_prompt=prompt)
  return _ok(passed=package_checks.passed(findings), findings=findings)


def render_submission(
    agent_name: str,
    analyzer_prompt: str,
    system_prompt_path: str,
    main_adapter_dir: str,
    tool_adapter_dir: str = "",
    temperature: float = 0.1,
    max_tokens: int = 8192,
    eval_n_val: int = 12,
    eval_seed: int = 0,
) -> str:
  """Renders the candidate submission package and audits it.

  The main system prompt is copied byte for byte from the training prompt, so
  it cannot drift from what the adapter was trained on. Adapter configs are
  copied with ``exclude_modules`` removed. When a tool adapter is given, a
  read-only ``code_analyzer`` sub-agent is rendered with it.

  Args:
    agent_name: Lowercase name of the agent, such as ``coder``.
    analyzer_prompt: Short, role-specific prompt for ``code_analyzer``.
    system_prompt_path: The training system prompt, relative to the working
      directory.
    main_adapter_dir: Directory holding the ``main_lora`` adapter files.
    tool_adapter_dir: Directory holding the ``tool_lora`` adapter files, or
      empty for a package without sub-agents.
    temperature: Sampling temperature, 0.0 to 0.3.
    max_tokens: Maximum generated tokens, at least 4096.
    eval_n_val: Validation episodes for the local evaluation.
    eval_seed: Seed of the validation shuffle.

  Returns:
    A JSON string with the candidate path and the audit result.
  """
  if not re.fullmatch(r"[a-z][a-z0-9_]*", agent_name):
    return _error("ValidationError", "agent_name must match [a-z][a-z0-9_]*")
  if tool_adapter_dir and not analyzer_prompt.strip():
    return _error("ValidationError", "analyzer_prompt must not be empty")
  if not 0.0 <= temperature <= package_checks.MAX_TEMPERATURE:
    return _error("ValidationError", f"temperature must be in [0, {package_checks.MAX_TEMPERATURE}]")
  if max_tokens < package_checks.MIN_MAX_TOKENS:
    return _error("ValidationError", f"max_tokens must be at least {package_checks.MIN_MAX_TOKENS}")
  try:
    system_prompt = _resolve(system_prompt_path)
    adapters = {package_checks.MAIN_ADAPTER: _resolve(main_adapter_dir)}
    if tool_adapter_dir:
      adapters[package_checks.TOOL_ADAPTER] = _resolve(tool_adapter_dir)
    out = _resolve(CANDIDATE_DIR)
  except _PathError as e:
    return _error("ValidationError", str(e))
  for path in [system_prompt] + [d / f for d in adapters.values() for f in ("adapter_config.json", "adapter_model.safetensors")]:
    if not path.is_file():
      return _error("FileNotFound", f"No such file: {path.relative_to(workdir())}")

  shutil.rmtree(out, ignore_errors=True)
  for name, src in adapters.items():
    dst = out / "adapters" / name
    dst.mkdir(parents=True)
    config = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    config.pop("exclude_modules", None)
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    shutil.copyfile(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
  (out / "prompts").mkdir()
  shutil.copyfile(system_prompt, out / "prompts" / "system.md")
  tools = ", ".join(package_checks.HARNESS_TOOLS)
  agent_yaml = (
      f"name: {agent_name}\n"
      "description: SWE agent fine-tuned on harness-format trajectories\n"
      f"model: {package_checks.COMPETITION_MODEL}\n"
      f"adapter: {package_checks.MAIN_ADAPTER}\n"
      "instruction: !include prompts/system.md\n"
      f"tools: [{tools}]\n"
  )
  if tool_adapter_dir:
    agent_yaml += "subagents: [code_analyzer]\n"
    (out / "prompts" / "analyzer.md").write_text(analyzer_prompt.strip() + "\n", encoding="utf-8")
    (out / "sub_agents").mkdir()
    (out / "sub_agents" / "code_analyzer.yaml").write_text(
        "name: code_analyzer\n"
        "description: Read-only sub-agent that inspects code and reports findings\n"
        f"model: {package_checks.COMPETITION_MODEL}\n"
        f"adapter: {package_checks.TOOL_ADAPTER}\n"
        "instruction: !include ../prompts/analyzer.md\n"
        f"tools: [{', '.join(package_checks.READ_ONLY_TOOLS)}]\n",
        encoding="utf-8",
    )
  (out / "agent.yaml").write_text(agent_yaml, encoding="utf-8")
  (out / "configs").mkdir()
  (out / "configs" / "sampling.yaml").write_text(
      f"temperature: {temperature}\nmax_tokens: {max_tokens}\n", encoding="utf-8"
  )
  (out / "eval_config.yaml").write_text(
      "# Candidate parameters. The schema is inferred; confirm it against\n"
      "# HARNESS_README.md and test it locally before submitting.\n"
      f"n_val: {eval_n_val}\nseed: {eval_seed}\n"
      "metrics: [next_tool_accuracy, val_loss]\n"
      f"score_thresholds: {list(spec.SCORE_THRESHOLDS)}\n",
      encoding="utf-8",
  )
  findings = package_checks.audit_package(out, training_prompt=system_prompt)
  return _ok(
      candidate=CANDIDATE_DIR,
      passed=package_checks.passed(findings),
      findings=findings,
  )
