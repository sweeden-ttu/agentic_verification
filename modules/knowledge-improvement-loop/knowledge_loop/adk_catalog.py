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

"""ADK components that could be added to the loop, checked against the install.

Each entry was found by introspecting google-adk 2.11.0 and reading adk.dev
(October 2026). ``catalog`` re-checks every import path at runtime, so the
list reports what the installed version really provides.
"""

from __future__ import annotations

import importlib
from typing import Any

# serves: a=evaluation, b=self-improvement/retry, c=parallel fan-out,
# d=human-in-the-loop, e=memory, f=eval tooling
CANDIDATES: list[dict[str, str]] = [
    {"name": "Workflow", "path": "google.adk.workflow:Workflow", "serves": "b,c",
     "use": "Root graph for research -> answer -> score -> route; replaces deprecated LoopAgent."},
    {"name": "JoinNode", "path": "google.adk.workflow:JoinNode", "serves": "c",
     "use": "Fan in parallel researcher branches before refinement."},
    {"name": "node", "path": "google.adk.workflow:node", "serves": "a,b",
     "use": "Wrap refine/CFR functions as graph nodes whose output picks the route."},
    {"name": "RetryConfig", "path": "google.adk.workflow:RetryConfig", "serves": "b",
     "use": "Back off and retry Claude nodes on overload."},
    {"name": "ReflectAndRetryToolPlugin", "path": "google.adk.plugins:ReflectAndRetryToolPlugin", "serves": "b",
     "use": "Retry tools whose JSON result has status=error, with reflection guidance (used here)."},
    {"name": "ReflectAndRetryModelPlugin", "path": "google.adk.plugins:ReflectAndRetryModelPlugin", "serves": "b",
     "use": "Re-prompt on malformed function calls or truncated output."},
    {"name": "FunctionTool(require_confirmation)", "path": "google.adk.tools:FunctionTool", "serves": "d",
     "use": "Human approval before an improvement is promoted (used here)."},
    {"name": "request_input", "path": "google.adk.tools:request_input", "serves": "d",
     "use": "Pause for the human to supply a missing score definition or file."},
    {"name": "SqliteMemoryService", "path": "google.adk.memory:SqliteMemoryService", "serves": "e",
     "use": "Persist scored answers and lessons across loop runs."},
    {"name": "App", "path": "google.adk.apps:App", "serves": "b,d,e",
     "use": "Holds plugins, compaction and resumability (used here)."},
    {"name": "AgentEvaluator", "path": "google.adk.evaluation:AgentEvaluator", "serves": "a,f",
     "use": "Regression evalsets for the loop's own behaviour."},
    {"name": "PrebuiltMetrics", "path": "google.adk.evaluation.eval_metrics:PrebuiltMetrics", "serves": "a,f",
     "use": "Rubric and LLM-judge metrics as the score; a Claude judge is untested."},
    {"name": "SimplePromptOptimizer", "path": "google.adk.optimization.simple_prompt_optimizer:SimplePromptOptimizer",
     "serves": "a,b", "use": "Evolve the coordinator prompt against evalset scores."},
    {"name": "ModelConsultTool", "path": "google.adk.tools:ModelConsultTool", "serves": "b",
     "use": "Let a cheaper model consult a stronger one before committing an answer."},
    {"name": "FallbackModel", "path": "google.adk.models:FallbackModel", "serves": "b",
     "use": "Fall through Claude model IDs on 529 overloads."},
    {"name": "RemoteA2aAgent", "path": "google.adk.agents.remote_a2a_agent:RemoteA2aAgent", "serves": "c",
     "use": "Call agents in other processes; needs google-adk[a2a]."},
]


def _installed(path: str) -> tuple[bool, str]:
  module, _, attr = path.partition(":")
  try:
    return hasattr(importlib.import_module(module), attr), ""
  except Exception as e:  # noqa: BLE001 - any import failure means not usable
    return False, f"{type(e).__name__}: {e}"[:160]


def catalog() -> list[dict[str, Any]]:
  """Every candidate with whether its import path resolves in this install."""
  out = []
  for c in CANDIDATES:
    ok, err = _installed(c["path"])
    out.append({**c, "installed": ok, **({"import_error": err} if err else {})})
  return out
