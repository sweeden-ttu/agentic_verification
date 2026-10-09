# LoRA improvement loop

The LoRA improvement loop is a Claude multi-agent team that verifies each run of a LoRA training notebook, plans the next run, and renders an audited submission package. A trace-language specification with a prophecy variable governs which steps the loop may take.

## Introduction

Training a LoRA adapter for a deadline involves a fixed budget of GPU sessions, a fixed deadline, and a package that a harness will reject for small mistakes. A model that plans the next session can plan one the budget does not allow, or report a score the notebook never produced. The loop removes that risk by checking every claim against a specification before it acts on the claim.

The loop has three layers. The `spec` module holds the transition relation and replays recorded traces. The `loop_tools` module holds the tools that read and write the working directory. The `agent` module holds the coordinator and three read-only specialists that call those tools.

The loop does not train and does not submit. The notebook trains, and a person submits, because only those two steps are expensive or irreversible.

## Get started

This example verifies a notebook run and plans the next job when the run beat the epoch-0 baseline. The tools return JSON strings so that they work as ADK function tools.

```python
import json

from lora_improvement_loop import loop_tools

verdict = json.loads(loop_tools.import_notebook_run("spec_state.json"))
if verdict["status"] == "ok" and verdict["metrics"]["improved_over_baseline"]:
  job = json.loads(
      loop_tools.plan_next_job(
          adapter="main_lora",
          learning_rate=1e-4,
          n_train=48,
          n_val=12,
          rationale="Validation loss fell for two epochs, so add data.",
      )
  )
```

To let Claude drive the same steps, point `adk web` at the package and describe the iteration in the first message. The coordinator is the module attribute `root_agent`.

```python
from lora_improvement_loop.agent import root_agent
```

## How it works

Each iteration starts from the notebook's `spec_state.json`. The coordinator calls `import_notebook_run`, which replays the trace and adopts it only when every step is allowed and the trace extends the one adopted before. The coordinator then asks `eval_designer` how to judge the next epoch and `improver` what to change. It calls `plan_next_job` with the values the improver gave, and the call succeeds only when the specification enables `TrainEpoch` today.

When the last run beat the baseline, the coordinator asks `submission_designer` for the analyzer prompt and calls `render_submission`. The tool copies the adapters, removes `exclude_modules` from their configs, copies the training prompt unchanged, writes `agent.yaml` and the `code_analyzer` sub-agent, and returns the result of `audit_package` for the new directory.

The specialists hold only `spec_status`, `list_artifacts` and `read_artifact`. They can read the run but cannot plan, render or import, so their advice reaches the files only through a coordinator tool call that validates it.

### The specification and the prophecy

A trace is valid when its first entry is the initial state and every later entry is the result of an enabled action. The `check_trace` function reports each step that is not. The same functions drive the live monitor, so the notebook and the loop cannot disagree about what is allowed.

The prophecy variable `p` predicts whether the next `Judge` action is a win or a loss. It selects which future the specification describes and does not cause that future. No guard reads it, `spec_status` does not show it, and `Judge` reports whether the outcome matched it. A test runs two behaviors that differ only in `p` and checks that they enable the same actions.

## Configuration options

The loop reads three environment variables and exposes the arguments of `plan_next_job` and `render_submission`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LOOP_WORKDIR` | path | `./work` | Directory that holds the notebook output, the jobs and the candidate package. |
| `LOOP_TODAY` | ISO date | the calendar date | Date used to catch the specification up with the schedule. |
| `LOOP_CLAUDE_MODEL` | string | `claude-opus-5-5` | Claude model for every agent in the team. |
| `learning_rate` | float | none | Learning rate of the next job, 5e-5 to 2e-4 for `main_lora` and 1e-5 to 1e-4 for `tool_lora`. |
| `n_train`, `n_val` | int | none | Training and validation trajectories, 8 to 5000 and 8 to 200. |
| `extra_exclude_repos` | list of strings | empty | Repositories in `owner__name` form to drop from the data, in addition to `encode__starlette`. |
| `temperature`, `max_tokens` | float, int | 0.1, 8192 | Sampling settings written to `configs/sampling.yaml`, with temperature at most 0.3 and at least 4096 tokens. |

`LOOP_WORKDIR` bounds every path a tool accepts. Absolute paths and `..` are refused, so a model cannot read or write outside the directory.

`LOOP_TODAY` exists so that tests and replays do not depend on the calendar. After the deadline the specification enables only `Judge`, so a loop run without this variable would refuse to plan anything.

The ranges on `plan_next_job` come from the project settings. The tool refuses values outside them instead of clamping, because a clamped value would silently differ from what the model reasoned about.

## Advanced applications

### Checking a trace without a model

When a notebook run looks wrong, replay its trace directly. This finds the first step the specification does not allow, which is often a forged or hand-edited state file.

```python
import json

from lora_improvement_loop import spec

state = json.load(open("spec_state.json"))
for problem in spec.check_trace(state["trace"]):
  print(problem)
```

### Auditing a package you built by hand

The audit does not depend on the agents. Run it on any directory that has the submission layout to see the checklist findings before you zip.

```python
from pathlib import Path

from lora_improvement_loop import package_checks

findings = package_checks.audit_package(
    Path("sample_submission"), training_prompt=Path("train_system.md")
)
print(package_checks.passed(findings), findings)
```

## Limitations

- The loop cannot train. Closing the loop requires running the notebook with the job's environment between agent runs.
- The notebook does not yet read `EXTRA_EXCLUDE_REPOS` and does not train `tool_lora`. A job that uses either needs a notebook change first.
- The sub-agent file format, the `subagents:` key and `eval_config.yaml` are inferred from the folder layout. The audit reports what it can check, and the harness documentation remains the authority.
- The audit reads safetensors headers only. It confirms key names and ranks and does not confirm that the weights load.
- The local score is a proxy. A passing audit and a better local score do not predict the leaderboard result.
- The tests script the model. They check the wiring and the guards, not the quality of Claude's advice.

## Related samples

- [SWE patch team](../../../../swe-patch-team/swe_patch_team/agent.py) - A coordinator with two read-only specialists wrapped as `AgentTool`s, which this team follows.
