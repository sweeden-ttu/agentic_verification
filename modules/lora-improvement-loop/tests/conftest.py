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

"""Shared fixtures: a working directory, notebook runs and a valid package."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from lora_improvement_loop import package_checks  # noqa: E402
from lora_improvement_loop import spec  # noqa: E402

SYSTEM_PROMPT = "You are a software engineer. Use the tools.\n"


@pytest.fixture
def work(tmp_path, monkeypatch):
  """A working directory on day 1 of the schedule (2026-10-09)."""
  directory = tmp_path / "work"
  directory.mkdir()
  monkeypatch.setenv("LOOP_WORKDIR", str(directory))
  monkeypatch.setenv("LOOP_TODAY", "2026-10-09")
  return directory


def write_notebook_run(work: Path, epochs: int = 1, name: str = "spec_state.json"):
  """Plays what the notebook does: epochs of TrainEpoch+Evaluate, then Package."""
  mon = spec.SpecMonitor(str(work / name), prophecy="win")
  mon.catch_up(spec.START.replace(day=9))
  rows = [{"epoch": 0, "val_loss": 2.0, "tool_acc": 0.2, "score": None}]
  for i in range(epochs):
    mon.train_epoch()
    acc = 0.3 + 0.2 * i
    mon.evaluate(spec.score_from_accuracy(acc))
    rows.append({"epoch": i + 1, "val_loss": 1.5 - 0.2 * i, "tool_acc": acc, "score": mon.s["score"]})
  (work / "metrics.json").write_text(json.dumps(rows))
  return mon


def write_safetensors(path: Path, r: int = 16, keys: list[str] | None = None) -> None:
  prefix = "base_model.model.model.language_model.layers.0."
  keys = keys or [
      prefix + "self_attn.q_proj.lora_A.weight",
      prefix + "self_attn.q_proj.lora_B.weight",
  ]
  header = {
      k: {
          "dtype": "BF16",
          "shape": [r, 64] if "lora_A" in k else [64, r],
          "data_offsets": [0, 0],
      }
      for k in keys
  }
  raw = json.dumps(header).encode()
  path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def write_adapter(folder: Path, r: int = 16, alpha: int = 32, **overrides) -> None:
  folder.mkdir(parents=True, exist_ok=True)
  config = {
      "r": r,
      "lora_alpha": alpha,
      "lora_dropout": 0.05,
      "target_modules": sorted(package_checks.TARGET_MODULES),
      "base_model_name_or_path": "gemma-4-31b-it-qat-q4_0-unquantized",
      **overrides,
  }
  config = {k: v for k, v in config.items() if v is not None}
  (folder / "adapter_config.json").write_text(json.dumps(config))
  write_safetensors(folder / "adapter_model.safetensors", r=r)


def make_package(root: Path, *, sub_agent: bool = True) -> Path:
  """Writes a package that passes every check."""
  root.mkdir(parents=True, exist_ok=True)
  write_adapter(root / "adapters" / "main_lora")
  (root / "prompts").mkdir()
  (root / "prompts" / "system.md").write_text(SYSTEM_PROMPT)
  (root / "configs").mkdir()
  (root / "configs" / "sampling.yaml").write_text("temperature: 0.1\nmax_tokens: 8192\n")
  (root / "eval_config.yaml").write_text("n_val: 12\n")
  agent = (
      "name: coder\ndescription: SWE agent\n"
      f"model: {package_checks.COMPETITION_MODEL}\nadapter: main_lora\n"
      "instruction: !include prompts/system.md\n"
      f"tools: [{', '.join(package_checks.HARNESS_TOOLS)}]\n"
  )
  if sub_agent:
    write_adapter(root / "adapters" / "tool_lora", r=8, alpha=16)
    (root / "prompts" / "analyzer.md").write_text("Explore code read-only and report findings.\n")
    (root / "sub_agents").mkdir()
    (root / "sub_agents" / "code_analyzer.yaml").write_text(
        "name: code_analyzer\ndescription: read-only\n"
        f"model: {package_checks.COMPETITION_MODEL}\nadapter: tool_lora\n"
        "instruction: !include ../prompts/analyzer.md\ntools: [read_file, get_status]\n"
    )
    agent += "subagents: [code_analyzer]\n"
  (root / "agent.yaml").write_text(agent)
  return root
