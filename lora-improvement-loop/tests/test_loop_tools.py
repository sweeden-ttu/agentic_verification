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

"""Tests for the loop tools."""

from __future__ import annotations

import json

from lora_improvement_loop import loop_tools as lt
from lora_improvement_loop import package_checks as pc
from lora_improvement_loop import spec
import pytest

from conftest import make_package, SYSTEM_PROMPT, write_adapter, write_notebook_run

JOB = dict(adapter="main_lora", learning_rate=1e-4, n_train=48, n_val=12, rationale="loss fell, raise data")


def _j(raw: str) -> dict:
  return json.loads(raw)


def test_spec_status_hides_the_prophecy(work):
  """The model never sees p: it is a ghost value."""
  status = _j(lt.spec_status())
  assert "p" not in status["state"] and status["enabled"]["TrainEpoch"]


def test_status_does_not_write_the_adopted_state(work):
  """Catching up to today's date happens in memory only."""
  write_notebook_run(work)
  lt.import_notebook_run("spec_state.json")
  before = (work / lt.ADOPTED_STATE).read_text()
  lt.spec_status()
  assert (work / lt.ADOPTED_STATE).read_text() == before


def test_import_adopts_a_verified_run_and_summarizes_metrics(work):
  """A valid notebook trace is adopted and compared with the baseline."""
  write_notebook_run(work, epochs=2)
  result = _j(lt.import_notebook_run("spec_state.json"))
  assert result["status"] == "ok" and "p" not in result["state"]
  assert result["metrics"]["improved_over_baseline"] is True
  assert (work / lt.ADOPTED_STATE).is_file()


def test_import_rejects_a_trace_outside_the_spec_language(work):
  """A forged extra epoch makes the import fail and adopts nothing."""
  write_notebook_run(work, epochs=3)
  state = json.loads((work / "spec_state.json").read_text())
  state["trace"].append({**state["trace"][-1], "action": "TrainEpoch", "phase": "eval", "budget": -1})
  (work / "forged.json").write_text(json.dumps(state))
  result = _j(lt.import_notebook_run("forged.json"))
  assert result["error_type"] == "TraceViolation"
  assert not (work / lt.ADOPTED_STATE).exists()


def test_import_rejects_a_valid_trace_that_rewrites_history(work):
  """A different but valid behavior does not extend the adopted trace."""
  write_notebook_run(work, epochs=1)
  lt.import_notebook_run("spec_state.json")
  other = spec.SpecMonitor(str(work / "other.json"), prophecy="win")
  other.catch_up(spec.START.replace(day=9))
  other.train_epoch()
  other.evaluate(3)
  assert spec.check_trace(other.s["trace"]) == []
  assert _j(lt.import_notebook_run("other.json"))["error_type"] == "TraceDiverged"


def test_import_accepts_a_longer_run_of_the_same_behavior(work):
  """A second import that extends the first is adopted."""
  write_notebook_run(work, epochs=1)
  lt.import_notebook_run("spec_state.json")
  mon = spec.SpecMonitor(str(work / "spec_state.json"))
  mon.train_epoch()
  mon.evaluate(3)
  assert _j(lt.import_notebook_run("spec_state.json"))["status"] == "ok"


@pytest.mark.parametrize("path", ["../outside.json", "/etc/passwd"])
def test_paths_cannot_leave_the_working_directory(work, path):
  """Traversal and absolute paths are refused."""
  assert _j(lt.import_notebook_run(path))["error_type"] == "ValidationError"
  assert _j(lt.read_artifact(path))["error_type"] == "ValidationError"


def test_plan_next_job_writes_the_environment_the_notebook_reads(work):
  """The job file carries LR, N_TRAIN, N_VAL and MAX_LEN for the notebook."""
  result = _j(lt.plan_next_job(**JOB))
  assert result["status"] == "ok"
  job = json.loads((work / result["job_path"]).read_text())
  assert job["env"]["LR"] == "0.0001" and job["env"]["MAX_LEN"] == "5120"
  assert job["locked"]["lora"] == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
  assert "p" not in job["spec_state"]


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 1e-2},
        {"adapter": "tool_lora", "learning_rate": 2e-4},
        {"n_train": 1},
        {"n_val": 1000},
        {"rationale": " "},
        {"extra_exclude_repos": ["not a repo"]},
        {"adapter": "other_lora"},
    ],
)
def test_plan_next_job_rejects_values_outside_the_allowed_ranges(work, change):
  """Locked and out-of-range settings cannot be planned."""
  result = _j(lt.plan_next_job(**{**JOB, **change}))
  assert result["error_type"] == "ValidationError"
  assert not (work / "jobs").exists()


def test_plan_next_job_waits_for_the_pending_job_to_run(work):
  """A second job is refused until the notebook has trained the first."""
  assert _j(lt.plan_next_job(**JOB))["status"] == "ok"
  assert _j(lt.plan_next_job(**JOB))["error_type"] == "JobPending"


def test_plan_next_job_is_refused_when_the_budget_is_spent(work):
  """With no weekly budget left the spec does not enable TrainEpoch."""
  write_notebook_run(work, epochs=spec.WEEKLY_BUDGET)
  lt.import_notebook_run("spec_state.json")
  assert _j(lt.plan_next_job(**JOB))["error_type"] == "SpecViolation"


def test_plan_next_job_is_refused_after_the_deadline(work, monkeypatch):
  """Past day 55 nothing can be planned."""
  monkeypatch.setenv("LOOP_TODAY", "2026-12-20")
  assert _j(lt.plan_next_job(**JOB))["error_type"] == "SpecViolation"


def _stage(work):
  write_adapter(work / "staged" / "main", exclude_modules=".*vision.*")
  write_adapter(work / "staged" / "tool", r=8, alpha=16)
  (work / "train_system.md").write_text(SYSTEM_PROMPT)


def test_render_submission_copies_the_training_prompt_and_passes_the_audit(work):
  """The candidate has the exact training prompt and no exclude_modules."""
  _stage(work)
  result = _j(lt.render_submission("coder", "Explore code read-only.", "train_system.md", "staged/main", "staged/tool"))
  assert result["passed"], result["findings"]
  candidate = work / lt.CANDIDATE_DIR
  assert (candidate / "prompts" / "system.md").read_bytes() == (work / "train_system.md").read_bytes()
  assert "exclude_modules" not in json.loads((candidate / "adapters" / "main_lora" / "adapter_config.json").read_text())
  assert "tools: [read_file, get_status]" in (candidate / "sub_agents" / "code_analyzer.yaml").read_text()


def test_render_submission_without_a_tool_adapter_has_no_sub_agents(work):
  """Leaving the tool adapter out renders a single-agent package."""
  _stage(work)
  result = _j(lt.render_submission("coder", "", "train_system.md", "staged/main"))
  assert result["passed"] and not (work / lt.CANDIDATE_DIR / "sub_agents").exists()


@pytest.mark.parametrize(
    "change",
    [{"agent_name": "Bad Name"}, {"temperature": 0.9}, {"max_tokens": 10}, {"system_prompt_path": "missing.md"}],
)
def test_render_submission_rejects_bad_inputs(work, change):
  """Invalid names, sampling values and missing files are refused."""
  _stage(work)
  args = dict(agent_name="coder", analyzer_prompt="Explore.", system_prompt_path="train_system.md", main_adapter_dir="staged/main", tool_adapter_dir="staged/tool")
  assert _j(lt.render_submission(**{**args, **change}))["status"] == "error"


def test_audit_package_tool_reports_findings(work):
  """The audit tool wraps the checks and reports pass or fail."""
  make_package(work / "pkg")
  assert _j(lt.audit_package("pkg"))["passed"] is True
  (work / "pkg" / "agent.yaml").write_text("name: x\n")
  assert _j(lt.audit_package("pkg"))["passed"] is False
