# agentic_verification
A framework that classifies autonomous AI agents by the Chomsky hierarchy, reduces unverifiable agents into verifiable ones through architectural interventions, and backs the whole thing with an epistemically self-aware knowledge base that knows what it does not know

## Layout

The repository is divided into three kinds of folder:

- **`modules/`** are reusable agent systems with tests and CI. Each one is a package another module or a notebook can import or run.
- **`experiments/`** are probes and studies. They produce runs, reports and findings rather than a component to reuse.
- **`utilities/`** are tools and data the other two depend on, including third-party code brought in as submodules.

## Modules

| Path | What it is |
|---|---|
| `modules/swe-patch-team/` | A three-agent Claude team (a coordinator plus read-only `code_analyzer` and `patch_reviewer`) that fixes an issue in a git checkout and submits a patch, using tools that mirror an offline SWE-bench-style harness. |
| `modules/lora-improvement-loop/` | A Claude multi-agent team that closes an improvement loop around the Gemma 4 LoRA training notebook. A trace-language spec (the notebook's TLA+ actions, with a ghost prophecy variable) verifies each run, three read-only specialists design the evaluation, the next training job and the submission's own sub-agents, and a deterministic audit checks the package against the pre-submit checklist. It does not train or submit. |
| `modules/knowledge-improvement-loop/` | A Claude multi-agent loop over a Kaggle evaluation baseline. Two researchers answer each question, a verifier votes, and the questions are refined to the ones with the most significant atoms and the most confident answers, each with follow-ups. Counterfactual regret minimization (CFR+) picks the change with the best guaranteed improvement across slices, and a person confirms it before it is promoted. |

## Experiments

| Path | What it is |
|---|---|
| `experiments/horizon-probe/` | Boundary analysis of where models diverge on the same question, with "legal" and "banned" as two separate axes. It has a 38-statement item bank built from the Gemma 4 Developer Agent forum and notebooks, and a recursive root question. Total surprise is split into within-model noise and between-model divergence (Bayes factors, per-model surprise). A stacked-RBM deep belief network and repo-local SWI-Prolog rules compute horizons, size pivots and universal definitions. Includes a CPU smoke run on Gemma 4 E4B and E2B. |
| `experiments/horizon-probe/data/kaggle_public_extract.{json,md}` | Rules, syntax, errors, task and test facts, allow/ban vocabulary and disputed questions extracted from the competition's public forum threads and notebooks, with sources |
| `experiments/kaggriculture/` | Two definitions of "legal", "wants", "shared borders" and "mutual success" for neighboring autonomous farmers, the $80 border underwriting as a Nash bargaining split, and an interactive farm-grid page |

## Utilities

| Path | What it is |
|---|---|
| `utilities/experimentally/` | Git submodule of [cloud-fronts-group/experimentally](https://github.com/cloud-fronts-group/experimentally), a fork of Evidently AI: an open-source framework to evaluate, test and monitor ML and LLM systems. |
| `utilities/datasets/gemma4-evidently-examples/` | Home for the Kaggle dataset `scottweeden/gemma4-evidently-examples`, the baseline `modules/knowledge-improvement-loop` reads. Run `fetch.sh` with Kaggle credentials to download it. |

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/sweeden-ttu/agentic_verification
# or, in an existing clone:
git submodule update --init
```
