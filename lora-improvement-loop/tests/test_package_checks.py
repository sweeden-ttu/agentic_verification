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

"""Tests for the pre-submit package audit."""

from __future__ import annotations

import json

from lora_improvement_loop import package_checks as pc
import pytest

from conftest import make_package, SYSTEM_PROMPT, write_adapter, write_safetensors


def _codes(findings, level="error"):
  return {f["code"] for f in findings if f["level"] == level}


def test_valid_package_passes(tmp_path):
  """A complete package with a matching training prompt has no errors."""
  root = make_package(tmp_path / "pkg")
  prompt = tmp_path / "train.md"
  prompt.write_text(SYSTEM_PROMPT)
  findings = pc.audit_package(root, training_prompt=prompt)
  assert pc.passed(findings), findings


def test_missing_training_prompt_is_only_a_warning(tmp_path):
  """Without the training prompt the match cannot be checked, so it warns."""
  findings = pc.audit_package(make_package(tmp_path / "pkg"))
  assert pc.passed(findings)
  assert "system_prompt_unverified" in _codes(findings, "warn")


def test_system_prompt_that_differs_from_training_is_an_error(tmp_path):
  """The main prompt must match the training prompt byte for byte."""
  root = make_package(tmp_path / "pkg")
  prompt = tmp_path / "train.md"
  prompt.write_text(SYSTEM_PROMPT + " ")
  assert "system_prompt" in _codes(pc.audit_package(root, training_prompt=prompt))


def test_exclude_modules_in_adapter_config_is_an_error(tmp_path):
  """The loader does not know exclude_modules, so it must be removed."""
  root = make_package(tmp_path / "pkg")
  write_adapter(root / "adapters" / "main_lora", exclude_modules=".*vision.*")
  assert "exclude_modules" in _codes(pc.audit_package(root))


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"lora_alpha": 64}, "main_lora_param"),
        ({"lora_dropout": 0.1}, "main_lora_param"),
        ({"target_modules": ["q_proj", "vision_proj"]}, "target_modules"),
        ({"base_model_name_or_path": "gemma-4-e4b-it"}, "wrong_base"),
    ],
)
def test_main_lora_settings_are_checked(tmp_path, overrides, code):
  """The main adapter must keep the project's fixed settings and 31B base."""
  root = make_package(tmp_path / "pkg")
  write_adapter(root / "adapters" / "main_lora", **overrides)
  assert code in _codes(pc.audit_package(root))


def test_unexpected_tensor_keys_are_an_error(tmp_path):
  """Tensors outside the language-model projections are rejected."""
  root = make_package(tmp_path / "pkg")
  write_safetensors(
      root / "adapters" / "main_lora" / "adapter_model.safetensors",
      keys=["vision_tower.layers.0.self_attn.q_proj.lora_A.weight"],
  )
  assert "tensor_keys" in _codes(pc.audit_package(root))


def test_tensor_rank_must_match_the_config(tmp_path):
  """A lora_A whose first dimension is not r is rejected."""
  root = make_package(tmp_path / "pkg")
  write_safetensors(root / "adapters" / "main_lora" / "adapter_model.safetensors", r=8)
  assert "tensor_rank" in _codes(pc.audit_package(root))


def test_wrong_model_and_unknown_adapter_are_errors(tmp_path):
  """agent.yaml must name the 31B model and an adapter folder that exists."""
  root = make_package(tmp_path / "pkg")
  text = (root / "agent.yaml").read_text()
  text = text.replace(pc.COMPETITION_MODEL, "gemma-4-e4b-it").replace("adapter: main_lora", "adapter: nope")
  (root / "agent.yaml").write_text(text)
  assert {"model", "adapter_name"} <= _codes(pc.audit_package(root))


def test_misspelled_tool_name_gets_a_suggestion(tmp_path):
  """An inexact tool name is an error that names the closest match."""
  root = make_package(tmp_path / "pkg")
  text = (root / "agent.yaml").read_text().replace("read_file", "read_files")
  (root / "agent.yaml").write_text(text)
  messages = [f["message"] for f in pc.audit_package(root) if f["code"] == "tool_name"]
  assert messages and "read_file" in messages[0]


def test_sub_agent_with_a_mutating_tool_is_an_error(tmp_path):
  """The analyzer may hold read-only tools only."""
  root = make_package(tmp_path / "pkg")
  path = root / "sub_agents" / "code_analyzer.yaml"
  path.write_text(path.read_text().replace("[read_file, get_status]", "[read_file, edit_file]"))
  assert "mutating_tool" in _codes(pc.audit_package(root))


def test_sub_agent_adapter_must_exist(tmp_path):
  """The sub-agent's adapter has to name a folder in adapters/."""
  root = make_package(tmp_path / "pkg")
  path = root / "sub_agents" / "code_analyzer.yaml"
  path.write_text(path.read_text().replace("tool_lora", "ghost_lora"))
  assert "adapter_name" in _codes(pc.audit_package(root))


def test_unresolved_include_is_an_error(tmp_path):
  """An !include target that does not exist is reported."""
  root = make_package(tmp_path / "pkg")
  (root / "prompts" / "analyzer.md").unlink()
  assert "include" in _codes(pc.audit_package(root))


def test_long_analyzer_prompt_warns(tmp_path):
  """A role-specific analyzer prompt stays short."""
  root = make_package(tmp_path / "pkg")
  (root / "prompts" / "analyzer.md").write_text("word " * 300)
  assert "analyzer_prompt_long" in _codes(pc.audit_package(root), "warn")


def test_sampling_outside_the_recommended_range_warns(tmp_path):
  """High temperature and small max_tokens are flagged, not blocked."""
  root = make_package(tmp_path / "pkg")
  (root / "configs" / "sampling.yaml").write_text("temperature: 0.9\nmax_tokens: 256\n")
  findings = pc.audit_package(root)
  assert pc.passed(findings)
  assert {"temperature", "max_tokens"} <= _codes(findings, "warn")


def test_excluded_repo_mention_warns(tmp_path):
  """A text file naming an excluded repo is flagged for a human to check."""
  root = make_package(tmp_path / "pkg")
  (root / "notes.md").write_text("trained on encode__starlette")
  assert "excluded_repo" in _codes(pc.audit_package(root), "warn")


def test_package_over_three_gib_is_an_error(tmp_path, monkeypatch):
  """The size limit is 3 GiB; the audit sums every file."""
  root = make_package(tmp_path / "pkg")
  monkeypatch.setattr(pc, "MAX_PACKAGE_BYTES", 100)
  assert "size" in _codes(pc.audit_package(root))


def test_missing_agent_yaml_stops_the_audit(tmp_path):
  """Without agent.yaml there is nothing else to check."""
  assert _codes(pc.audit_package(tmp_path)) == {"layout"}


def test_corrupt_safetensors_header_is_reported(tmp_path):
  """A truncated safetensors file yields an error instead of an exception."""
  root = make_package(tmp_path / "pkg")
  (root / "adapters" / "main_lora" / "adapter_model.safetensors").write_bytes(b"abc")
  assert "safetensors_header" in _codes(pc.audit_package(root))
