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

import json
import os
from typing import Any, Optional

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.anthropic_llm import Claude
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

from . import loop_tools

_MODEL = Claude(
    model=os.environ.get("LOOP_CLAUDE_MODEL", "claude-opus-5-5"),
    max_tokens=16384,
)

_READ_ONLY = [
    loop_tools.adk_candidates,
    loop_tools.discover_baseline,
    loop_tools.read_workdir_file,
]

researcher_empirical = Agent(
    model=_MODEL,
    name="researcher_empirical",
    description="Answers one question from data: baseline tables, measured numbers, installed ADK components.",
    instruction="""
You answer one research question from data only.
- Use discover_baseline and read_workdir_file for numbers, adk_candidates for
  what the installed ADK provides.
- Every claim cites a file, a column and a number, or an import path.
- If the data cannot answer it, say "unsupported" and name the missing data.
Reply in under 150 words.
""",
    tools=list(_READ_ONLY),
)

researcher_skeptic = Agent(
    model=_MODEL,
    name="researcher_skeptic",
    description="Answers the same question by looking for the reasons the obvious answer is wrong.",
    instruction="""
You answer one research question by testing the obvious answer against the
data: look for slices where it fails, confounds, and missing baselines.
- Use the same read-only tools and cite file, column and number.
- State the answer you end up with, not only the objections.
Reply in under 150 words.
""",
    tools=list(_READ_ONLY),
)

verifier = Agent(
    model=_MODEL,
    name="verifier",
    description="Checks each recorded answer's evidence against the files and says supported or not.",
    instruction="""
You verify answers. For the question id you are given, read nodes.json with
read_workdir_file. For each answer, check its cited numbers and import paths
with the read-only tools. Reply with one line per answer index:
"<index>: SUPPORTED" or "<index>: UNSUPPORTED - <reason>". Do not judge style.
""",
    tools=list(_READ_ONLY),
)

root_agent = Agent(
    model=_MODEL,
    name="knowledge_coordinator",
    description="Runs one round of the knowledge-improvement loop over a Kaggle evaluation baseline.",
    instruction="""
You run one round of a knowledge-improvement loop. Numbers come from tools.

1. Call discover_baseline. If it errors, report that the baseline is missing
   and continue with the ADK questions only.
2. Pose at most 3 questions about what would most improve the baseline score
   (one may be which ADK component to add: see adk_candidates).
3. For each question, ask researcher_empirical and researcher_skeptic, and save
   each reply with record_answer (branch "empirical" or "skeptic").
4. Ask verifier about each question and save every verdict with record_vote.
5. Call refine_questions. Keep only questions it keeps; their follow-ups are
   the next round's questions.
6. If the baseline has a table with candidate, slice and score columns, call
   robust_improvement. Prefer its robust policy over the best-on-average
   candidate, and report both.
7. Call promote_improvement only for a change with a positive guaranteed
   improvement. A human confirms it before it is recorded.

Finish with: kept questions with their best answer and confidence, the
follow-ups, the robust choice, and what was promoted or refused.
""",
    tools=[
        AgentTool(researcher_empirical),
        AgentTool(researcher_skeptic),
        AgentTool(verifier),
        loop_tools.adk_candidates,
        loop_tools.discover_baseline,
        loop_tools.record_answer,
        loop_tools.record_vote,
        loop_tools.refine_questions,
        loop_tools.robust_improvement,
        FunctionTool(loop_tools.promote_improvement, require_confirmation=True),
    ],
)


class JsonErrorRetryPlugin(ReflectAndRetryToolPlugin):
  """Treats a ``{"status": "error"}`` JSON tool result as a failure to reflect on."""

  async def extract_error_from_result(self, *, tool, tool_args, tool_context, result) -> Optional[dict[str, Any]]:
    text = result.get("result") if isinstance(result, dict) else result
    if not isinstance(text, str):
      return None
    try:
      payload = json.loads(text)
    except ValueError:
      return None
    if isinstance(payload, dict) and payload.get("status") == "error":
      return payload
    return None


app = App(
    name="knowledge_loop",
    root_agent=root_agent,
    plugins=[JsonErrorRetryPlugin(max_retries=2, throw_exception_if_retry_exceeded=False)],
)
