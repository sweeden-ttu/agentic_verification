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

import os

from google.adk import Agent
from google.adk.models.anthropic_llm import Claude
from google.adk.tools.agent_tool import AgentTool

from . import workspace_tools

# One Claude model backs every agent in the team, as in a single-base-model
# SWE harness. Override with SWE_CLAUDE_MODEL.
_MODEL = Claude(
    model=os.environ.get("SWE_CLAUDE_MODEL", "claude-opus-5-5"),
    max_tokens=16384,
)

# Read-only specialist. Wrapped as an AgentTool so its file dumps stay out of
# the coordinator's context window.
code_analyzer = Agent(
    model=_MODEL,
    name="code_analyzer",
    description=(
        "Read-only code explorer. Give it a question about the repository and"
        " it returns the relevant files, line ranges and a short explanation."
    ),
    instruction="""
You explore a repository and answer one question about it. You cannot modify
anything.
- Locate code with list_files and search_code, then read the exact ranges you
  need with read_file (at most 150 lines per call).
- Reply with: the files and line ranges that matter, how they relate, and the
  most likely place to change. Quote only short snippets.
- Keep the reply under 300 words. Do not propose a full implementation.
""",
    tools=[
        workspace_tools.list_files,
        workspace_tools.search_code,
        workspace_tools.read_file,
    ],
)

# Read-only second opinion on the working-tree diff.
patch_reviewer = Agent(
    model=_MODEL,
    name="patch_reviewer",
    description=(
        "Reviews the current uncommitted diff against the task description you"
        " pass in. Returns APPROVE or REQUEST_CHANGES with specific issues."
    ),
    instruction="""
You review a proposed patch. The request contains the task description.
1. Call get_diff and read it fully.
2. Check that it fixes the described problem at its root, does not touch test
   files or runner config (conftest.py, pytest.ini, pyproject.toml, tox.ini,
   setup.cfg), and does not include scratch or reproduction files.
3. Use read_file / search_code for surrounding context when the diff is not
   enough, for example to find other callers of a changed function.
Answer with a first line of exactly APPROVE or REQUEST_CHANGES, followed by a
short bullet list of concrete issues (file and line) if any.
""",
    tools=[
        workspace_tools.get_diff,
        workspace_tools.read_file,
        workspace_tools.search_code,
    ],
)

root_agent = Agent(
    model=_MODEL,
    name="swe_coordinator",
    description="Fixes a software issue in a repository and submits a patch.",
    instruction="""
You are a software engineer fixing an issue in the repository checked out in
the workspace. The user message holds the problem statement.

Workflow:
1. Ask code_analyzer where the problem lives. Delegate broad exploration to it
   instead of reading many files yourself.
2. Make the smallest correct change to library code with edit_file (or
   write_file for new files). Prefer several small edits over one large one.
3. Verify with run_command: run the relevant existing tests, or a reproduction
   script written under /tmp (never inside the workspace).
4. Ask patch_reviewer to review the diff, passing it the problem statement.
   Fix real issues it raises, then re-verify.
5. Delete any scratch files from the workspace, then call submit_patch as your
   final action, and finish with a one-paragraph summary.

Rules:
- Never edit test files or test runner configuration; fix the library code.
- The environment is offline: do not pip install anything.
- Keep your reasoning short and make tool calls promptly.
""",
    tools=[
        AgentTool(code_analyzer),
        AgentTool(patch_reviewer),
        workspace_tools.run_command,
        workspace_tools.read_file,
        workspace_tools.edit_file,
        workspace_tools.write_file,
        workspace_tools.submit_patch,
    ],
)
