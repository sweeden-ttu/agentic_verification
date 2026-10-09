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

"""Tests for the swe_patch_team package."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import subprocess
import sys

from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from swe_patch_team import agent as swe_agent  # noqa: E402
from swe_patch_team import workspace_tools  # noqa: E402


def _git(repo: Path, *args: str) -> None:
  subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
  (tmp_path / "calc.py").write_text(
      "def add(a, b):\n    return a - b\n", encoding="utf-8"
  )
  _git(tmp_path, "init", "-q")
  _git(tmp_path, "config", "user.email", "scott.weeden@gmail.com")
  _git(tmp_path, "config", "user.name", "sweeden-tty")
  _git(tmp_path, "add", "-A")
  _git(tmp_path, "commit", "-qm", "baseline")
  monkeypatch.setenv("SWE_WORKSPACE", str(tmp_path))
  return tmp_path


def test_read_file_rejects_traversal(repo):
  result = json.loads(workspace_tools.read_file("../etc/passwd"))
  assert result["status"] == "error"
  assert result["error_type"] == "ValidationError"


def test_read_file_strips_workspace_prefix_and_caps_lines(repo):
  (repo / "big.txt").write_text("\n".join(map(str, range(400))))
  result = json.loads(workspace_tools.read_file("/workspace/big.txt"))
  assert result["end_line"] == 150
  assert result["total_lines"] == 400
  assert result["is_truncated"] is True


def test_edit_file_exact_and_flexible(repo):
  exact = json.loads(
      workspace_tools.edit_file("calc.py", "return a - b", "return a + b")
  )
  assert (exact["status"], exact["strategy"]) == ("ok", "exact")

  (repo / "cls.py").write_text("class C:\n    def f(self):\n        return 1\n")
  flexible = json.loads(
      workspace_tools.edit_file(
          "cls.py", "def f(self):\n    return 1", "def f(self):\n    return 2"
      )
  )
  assert (flexible["status"], flexible["strategy"]) == ("ok", "flexible")
  assert (repo / "cls.py").read_text() == (
      "class C:\n    def f(self):\n        return 2\n"
  )


def test_edit_file_ambiguous_match_is_an_error(repo):
  (repo / "dup.py").write_text("x = 1\nx = 1\n")
  result = json.loads(workspace_tools.edit_file("dup.py", "x = 1", "x = 2"))
  assert result["error_type"] == "FileEditError"
  assert (repo / "dup.py").read_text() == "x = 1\nx = 1\n"


async def test_run_command_reports_exit_code(repo):
  ok = json.loads(await workspace_tools.run_command("echo hi"))
  assert ok["stdout"] == "hi\n"
  bad = json.loads(await workspace_tools.run_command("exit 3"))
  assert bad["error_type"] == "CommandError"
  assert bad["details"]["exit_code"] == 3


async def test_team_fixes_bug_and_submits_patch(repo, monkeypatch):
  """Scripts Claude for all three agents and checks the delegation chain."""
  calls = collections.Counter()

  def call(name: str, **args) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
        )
    )

  def say(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )

  async def scripted(self, llm_request, stream=False):
    instruction = str(llm_request.config.system_instruction)
    if "You are a software engineer" in instruction:
      role = "coordinator"
    elif "explore a repository" in instruction:
      role = "analyzer"
    else:
      role = "reviewer"
    step = calls[role]
    calls[role] += 1
    script = {
        "coordinator": [
            call("code_analyzer", request="Where is add() defined?"),
            call("edit_file", filepath="calc.py", old_string="a - b", new_string="a + b"),
            call("patch_reviewer", request="add() subtracts instead of adding."),
            call("submit_patch"),
            say("Fixed add()."),
        ],
        "analyzer": [call("search_code", pattern="def add"), say("calc.py:1")],
        "reviewer": [call("get_diff"), say("APPROVE")],
    }[role]
    yield script[step]

  monkeypatch.setattr(AnthropicLlm, "generate_content_async", scripted)

  runner = InMemoryRunner(agent=swe_agent.root_agent, app_name="swe_test")
  session = await runner.session_service.create_session(
      app_name="swe_test", user_id="u"
  )
  message = types.Content(role="user", parts=[types.Part(text="add() is wrong")])
  async for _ in runner.run_async(
      user_id="u", session_id=session.id, new_message=message
  ):
    pass

  assert calls == {"coordinator": 5, "analyzer": 2, "reviewer": 2}
  final = await runner.session_service.get_session(
      app_name="swe_test", user_id="u", session_id=session.id
  )
  assert final.state[workspace_tools.PATCH_SUBMITTED_KEY] is True
  assert "+    return a + b" in final.state[workspace_tools.SUBMITTED_PATCH_KEY]
