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

from . import loop_tools

# One Claude model backs every LLM agent in the loop. Override with
# LOOP_CLAUDE_MODEL. The training itself runs elsewhere, on the baseline
# notebook's GPU runtime.
_MODEL = Claude(
    model=os.environ.get("LOOP_CLAUDE_MODEL", "claude-opus-5-5"),
    max_tokens=16384,
)

# Read-only specialist: designs the evaluation the next epoch is judged by.
eval_designer = Agent(
    model=_MODEL,
    name="eval_designer",
    description=(
        "Designs the evaluation for the next training epoch from the metrics"
        " history: validation size, thresholds and what counts as improvement."
    ),
    instruction="""
You design evaluations for a LoRA adapter that picks the next tool of an SWE
agent (run_command, read_file, edit_file, write_file, submit_patch).
- Call spec_status, then read metrics.json with read_artifact.
- The score is teacher-forced next-tool accuracy mapped to 0..3 by thresholds
  0.25, 0.50 and 0.75, with validation loss on model tokens alongside.
- State: how many validation trajectories to use (8 to 200) and why, whether
  the thresholds still separate good from bad epochs, and the exact condition
  that counts as improvement over the epoch-0 baseline.
- Tune against the validation episodes. Do not guess. Cite the numbers you
  read. Keep the reply under 200 words.
""",
    tools=[
        loop_tools.spec_status,
        loop_tools.list_artifacts,
        loop_tools.read_artifact,
    ],
)

# Read-only specialist: proposes the next training job.
improver = Agent(
    model=_MODEL,
    name="improver",
    description=(
        "Reads the metrics history and the package audit, then proposes one"
        " change to the next training job with its reason."
    ),
    instruction="""
You propose the single most useful change to the next LoRA training job.
- Read metrics.json and the latest audit with read_artifact and spec_status.
- Locked and not yours to change: the 31B base, rank 16, alpha 32, dropout
  0.05, gradient accumulation 8, max length 5120, language-model-only targets,
  loss on model tokens only.
- You may change: adapter (main_lora or tool_lora), learning rate (main_lora
  5e-5 to 2e-4, tool_lora 1e-5 to 1e-4), n_train (8 to 5000), n_val (8 to 200),
  and extra repos to exclude from the data.
- If validation loss rose while accuracy fell, lower the learning rate. If both
  improved, raise n_train. If TrainEpoch is not enabled, say to wait.
- Reply with the values and a two-sentence reason that cites the metrics.
""",
    tools=[
        loop_tools.spec_status,
        loop_tools.list_artifacts,
        loop_tools.read_artifact,
    ],
)

# Read-only specialist: designs the multi-agent submission.
submission_designer = Agent(
    model=_MODEL,
    name="submission_designer",
    description=(
        "Designs the submission agent system: the read-only code_analyzer"
        " sub-agent prompt and the sampling settings."
    ),
    instruction="""
You design the multi-agent shape of a Gemma 4 SWE submission: one main agent
plus a read-only code_analyzer sub-agent.
- The analyzer prompt is short and role-specific: under 150 words, it explores
  code with read_file and get_status only and reports files, line ranges and
  findings. It never edits.
- Sampling: temperature 0.0 to 0.3 and a large max_tokens (at least 4096).
- Do not write or change the main system prompt. It is copied from training.
- Reply with the analyzer prompt text, the temperature and max_tokens.
""",
    tools=[
        loop_tools.spec_status,
        loop_tools.list_artifacts,
        loop_tools.read_artifact,
    ],
)

root_agent = Agent(
    model=_MODEL,
    name="loop_coordinator",
    description=(
        "Runs one iteration of the LoRA improvement loop for a Gemma 4 SWE"
        " agent: verify the last notebook run, plan the next job, and render"
        " the best candidate package."
    ),
    instruction="""
You run one iteration of a self-improvement loop around a LoRA training
notebook. The notebook trains; you verify, plan and package. The loop is
governed by a spec, and spec_status shows which actions are enabled today.

Iteration:
1. Call spec_status.
2. If the user names a notebook spec_state.json, call import_notebook_run. A
   TraceViolation or TraceDiverged error means the run cannot be trusted: stop
   and report the problems. Never edit the spec files.
3. Ask eval_designer for the evaluation and improver for the next change.
4. If TrainEpoch is enabled, call plan_next_job with the improver's values. If
   it is not, say when the budget refills and do not plan.
5. If the improved_over_baseline flag is true and the user names the staged
   adapters, ask submission_designer for the analyzer prompt, then call
   render_submission with those values. The system prompt comes from the
   training prompt path the user gives you; never write it yourself.
6. Call audit_package on the candidate and report every error and warning.

Rules:
- You cannot train, submit or judge. Training runs in the notebook; submitting
  is done by a human, who zips the package from inside its directory.
- Never claim the leaderboard score will improve. The local score is a proxy.
- Finish with a short summary: spec state, the job you planned, the audit
  result, and what the human must run next.
""",
    tools=[
        AgentTool(eval_designer),
        AgentTool(improver),
        AgentTool(submission_designer),
        loop_tools.spec_status,
        loop_tools.import_notebook_run,
        loop_tools.plan_next_job,
        loop_tools.render_submission,
        loop_tools.audit_package,
    ],
)
