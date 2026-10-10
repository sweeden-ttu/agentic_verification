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

"""Runs the coordinator with a scripted Claude."""

from __future__ import annotations

import collections
import json

from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

from knowledge_loop import agent as loop_agent
from knowledge_loop import loop_tools


def _call(name, **args):
  return LlmResponse(content=types.Content(role="model", parts=[
      types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def _say(text):
  return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _role(instruction):
  for marker, role in (("from data only", "empirical"), ("testing the obvious", "skeptic"), ("You verify answers", "verifier")):
    if marker in instruction:
      return role
  return "coordinator"


async def _run(monkeypatch, scripts, use_app=False):
  calls = collections.Counter()

  async def scripted(self, llm_request, stream=False):
    role = _role(str(llm_request.config.system_instruction))
    step = calls[role]
    calls[role] += 1
    yield scripts[role][min(step, len(scripts[role]) - 1)]

  monkeypatch.setattr(AnthropicLlm, "generate_content_async", scripted)
  runner = InMemoryRunner(app=loop_agent.app) if use_app else InMemoryRunner(agent=loop_agent.root_agent, app_name="t")
  app_name = runner.app_name
  session = await runner.session_service.create_session(app_name=app_name, user_id="u")
  msg = types.Content(role="user", parts=[types.Part(text="Run one round.")])
  events = [e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=msg)]
  return calls, events


@pytest.fixture
def work(tmp_path, monkeypatch):
  monkeypatch.setenv("LOOP_WORKDIR", str(tmp_path / "work"))
  monkeypatch.setenv("KAGGLE_BASELINE_DIR", str(tmp_path / "missing"))
  return tmp_path / "work"


async def test_round_records_both_branches_votes_and_refines(work, monkeypatch):
  """Coordinator asks both researchers and the verifier, records, then refines."""
  scripts = {
      "coordinator": [
          _call("discover_baseline"),
          _call("researcher_empirical", request="Which ADK component helps most?"),
          _call("record_answer", question_id="q1", question="Which ADK component?", branch="empirical", answer="add reflect retry plugin"),
          _call("researcher_skeptic", request="Which ADK component helps most?"),
          _call("record_answer", question_id="q1", question="Which ADK component?", branch="skeptic", answer="add sqlite memory service"),
          _call("verifier", request="Verify q1"),
          _call("record_vote", question_id="q1", answer_index=0, supported=True),
          _call("refine_questions", top_k=3),
          _say("Round done."),
      ],
      "empirical": [_call("adk_candidates"), _say("add reflect retry plugin")],
      "skeptic": [_say("add sqlite memory service")],
      "verifier": [_call("read_workdir_file", filepath="nodes.json"), _say("0: SUPPORTED")],
  }
  calls, _ = await _run(monkeypatch, scripts)
  nodes = json.loads((work / "nodes.json").read_text())
  assert [a["branch"] for a in nodes[0]["answers"]] == ["empirical", "skeptic"]
  assert nodes[0]["answers"][0]["votes"] == [True]
  assert calls["coordinator"] == 9 and calls["verifier"] == 2


async def test_promotion_pauses_for_human_confirmation(work, monkeypatch):
  """promote_improvement asks for confirmation and writes nothing yet."""
  scripts = {"coordinator": [
      _call("promote_improvement", title="Use steady prompt", evidence="+0.05 worst slice", expected_gain=0.05),
      _say("Waiting for approval."),
  ]}
  _, events = await _run(monkeypatch, scripts)
  names = [p.function_call.name for e in events for p in (e.content.parts if e.content else []) if p.function_call]
  assert "adk_request_confirmation" in names
  assert not (work / "improvements.json").exists()


async def test_json_error_results_are_retried_with_reflection(work, monkeypatch):
  """Under the App, a status=error tool result reaches the model as a retry prompt."""
  seen = []
  scripts = {"coordinator": [
      _call("record_vote", question_id="nope", answer_index=0, supported=True),
      _say("ok"),
  ]}
  orig = loop_agent.JsonErrorRetryPlugin.extract_error_from_result

  async def spy(self, **kw):
    out = await orig(self, **kw)
    seen.append(out)
    return out

  monkeypatch.setattr(loop_agent.JsonErrorRetryPlugin, "extract_error_from_result", spy)
  await _run(monkeypatch, scripts, use_app=True)
  assert seen and seen[0] and seen[0]["error_type"] == "NotFound"


def test_specialists_cannot_write():
  """Researchers and verifier only hold read-only tools."""
  writers = {loop_tools.record_answer, loop_tools.record_vote, loop_tools.promote_improvement, loop_tools.refine_questions}
  for sub in (loop_agent.researcher_empirical, loop_agent.researcher_skeptic, loop_agent.verifier):
    assert not writers & set(sub.tools)
