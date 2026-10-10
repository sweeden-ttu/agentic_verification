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

"""Runs the coordinator and its specialists with a scripted Claude."""

from __future__ import annotations

import collections
import json

from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from conftest import SYSTEM_PROMPT, write_adapter, write_notebook_run
from lora_improvement_loop import agent as loop_agent


def _call(name: str, **args) -> LlmResponse:
  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
      )
  )


def _say(text: str) -> LlmResponse:
  return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _role(instruction: str) -> str:
  for marker, role in (
      ("You design evaluations", "eval_designer"),
      ("You propose the single most useful", "improver"),
      ("You design the multi-agent shape", "submission_designer"),
  ):
    if marker in instruction:
      return role
  return "coordinator"


async def _run(monkeypatch, scripts, text="Run one iteration."):
  calls = collections.Counter()

  async def scripted(self, llm_request, stream=False):
    role = _role(str(llm_request.config.system_instruction))
    step = calls[role]
    calls[role] += 1
    yield scripts[role][step]

  monkeypatch.setattr(AnthropicLlm, "generate_content_async", scripted)
  runner = InMemoryRunner(agent=loop_agent.root_agent, app_name="loop_test")
  session = await runner.session_service.create_session(app_name="loop_test", user_id="u")
  message = types.Content(role="user", parts=[types.Part(text=text)])
  tool_results = []
  async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message):
    for part in event.content.parts if event.content else []:
      if part.function_response:
        tool_results.append((part.function_response.name, part.function_response.response))
  return calls, tool_results


async def test_iteration_verifies_the_run_plans_a_job_and_renders_a_candidate(work, monkeypatch):
  """One iteration: import, ask three specialists, plan, render, audit."""
  write_notebook_run(work, epochs=2)
  (work / "train_system.md").write_text(SYSTEM_PROMPT)
  write_adapter(work / "staged" / "main")
  write_adapter(work / "staged" / "tool", r=8, alpha=16)
  scripts = {
      "coordinator": [
          _call("spec_status"),
          _call("import_notebook_run", spec_state_path="spec_state.json"),
          _call("eval_designer", request="Design the next evaluation."),
          _call("improver", request="Propose the next change."),
          _call("plan_next_job", adapter="main_lora", learning_rate=1e-4, n_train=48, n_val=12, rationale="val loss fell"),
          _call("submission_designer", request="Design the analyzer."),
          _call(
              "render_submission",
              agent_name="coder",
              analyzer_prompt="Explore code read-only.",
              system_prompt_path="train_system.md",
              main_adapter_dir="staged/main",
              tool_adapter_dir="staged/tool",
          ),
          _call("audit_package", package_dir="candidate_submission", training_prompt_path="train_system.md"),
          _say("Job planned; candidate rendered."),
      ],
      "eval_designer": [_call("read_artifact", filepath="metrics.json"), _say("Use 12 episodes.")],
      "improver": [_call("spec_status"), _say("Raise n_train to 48.")],
      "submission_designer": [_say("Prompt: explore, report.")],
  }
  calls, results = await _run(monkeypatch, scripts)
  audit = json.loads(dict(results)["audit_package"]["result"])
  assert calls["coordinator"] == 9 and calls["eval_designer"] == 2
  assert (work / "jobs" / "job_001.json").is_file()
  assert (work / "candidate_submission" / "agent.yaml").is_file()
  assert audit["passed"] is True, audit["findings"]


async def test_planning_past_the_budget_is_refused_even_if_the_model_asks(work, monkeypatch):
  """The spec, not the prompt, stops a fourth epoch in the same week."""
  write_notebook_run(work, epochs=3)
  scripts = {
      "coordinator": [
          _call("import_notebook_run", spec_state_path="spec_state.json"),
          _call("plan_next_job", adapter="main_lora", learning_rate=1e-4, n_train=48, n_val=12, rationale="try again"),
          _say("Budget spent; wait for the refill."),
      ],
  }
  _, results = await _run(monkeypatch, scripts)
  assert "SpecViolation" in json.dumps(dict(results)["plan_next_job"])
  assert not (work / "jobs").exists()


async def test_a_forged_notebook_trace_is_not_adopted(work, monkeypatch):
  """The coordinator receives TraceViolation and nothing is adopted."""
  write_notebook_run(work, epochs=3)
  state = json.loads((work / "spec_state.json").read_text())
  state["trace"][-1]["best"] = 0
  (work / "spec_state.json").write_text(json.dumps(state))
  scripts = {
      "coordinator": [
          _call("import_notebook_run", spec_state_path="spec_state.json"),
          _say("The run cannot be trusted."),
      ]
  }
  _, results = await _run(monkeypatch, scripts)
  assert "TraceViolation" in json.dumps(dict(results)["import_notebook_run"])
  assert not (work / "adopted_spec_state.json").exists()


def test_specialists_hold_only_read_only_tools():
  """The three sub-agents cannot plan, render or import."""
  allowed = {"spec_status", "list_artifacts", "read_artifact"}
  for sub in (loop_agent.eval_designer, loop_agent.improver, loop_agent.submission_designer):
    assert {t.__name__ for t in sub.tools} == allowed


def test_no_tool_can_submit_or_judge():
  """Submit and Judge are human steps: no coordinator tool is named for them."""
  names = {getattr(t, "__name__", getattr(t, "name", "")) for t in loop_agent.root_agent.tools}
  assert not {n for n in names if "submit" in n.lower() and n != "render_submission"}
  assert not {n for n in names if "judge" in n.lower()}
