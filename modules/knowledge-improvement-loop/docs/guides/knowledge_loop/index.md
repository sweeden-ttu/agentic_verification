# Knowledge improvement loop

The knowledge improvement loop is a Claude multi-agent team that refines a graph of research questions and picks the change with the best guaranteed improvement on an evaluation score. It combines a Bayesian decomposition of answers into atoms with counterfactual regret minimization over evaluation slices.

## Introduction

Improving a baseline needs two decisions: what to ask next, and which change to adopt. A question is worth keeping when independent researchers disagree on it in a specific way and one answer holds up under verification. A change is worth adopting when it improves the score on every slice the evaluation measures, not only on average. The `refine` module answers the first decision and the `baseline` and `cfr` modules answer the second.

## Get started

This example refines two recorded questions and solves for the robust candidate on a baseline table.

```python
from knowledge_loop import baseline, refine

nodes = [
    {"id": "q1", "question": "Which adapter?", "answers": [
        {"branch": "empirical", "text": "use lora rank sixteen", "votes": [True, True, True]},
        {"branch": "empirical", "text": "use lora rank sixteen", "votes": []},
        {"branch": "skeptic", "text": "use full finetune", "votes": []},
        {"branch": "skeptic", "text": "use full finetune", "votes": []},
    ]},
]
kept = refine.refine(nodes, top_k=3)["kept"]

rows = baseline.read_rows(baseline.baseline_root() / "eval.csv")
choice = baseline.robust_choice(rows, candidate_col="prompt", slice_col="task", score_col="score", baseline="v0")
```

## How it works

Every answer is reduced to atoms. For each atom and each branch, the module counts the answers that contain it and the answers that do not, and it splits the total surprise into noise within a branch and mutual information with the branch. An atom with enough mutual information and a Bayes factor above the gate marks a real disagreement. A question's value multiplies the total information of its significant atoms by the confidence of its best answer, so a question ranks high only when it both separates the branches and has a trustworthy answer.

Follow-ups are written for the significant atoms of each kept question. They rank by the atom's information times the uncertainty of the best answer, because a follow-up pays off most where the disagreement is large and the answer is still open.

The baseline table becomes a matrix of mean improvement over the baseline candidate for every candidate and slice. CFR+ solves it as a zero-sum game in which an adversary chooses the slice. The row policy is the candidate mix with the highest guaranteed improvement, and the exploitability shows how close the policy is to an equilibrium.

## Configuration options

The loop reads its locations from environment variables, and the refinement gates are arguments.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `LOOP_WORKDIR` | path | `./work` | Holds `nodes.json`, `refined.json` and `improvements.json`. |
| `KAGGLE_BASELINE_DIR` | path | the Kaggle input path | Root of the baseline tables. |
| `LOOP_CLAUDE_MODEL` | string | `claude-opus-5-5` | Model for every agent. |
| `mi_min_bits` | float | `0.1` | Least mutual information for a significant atom. |
| `log_bf_min` | float | `1.0` | Least natural-log Bayes factor for a significant atom. |
| `top_k` | int | `5` | Questions kept per round. |

The two gates come from `horizon-probe/config/run.json`, so results compare with that probe. Raising them keeps fewer atoms and fewer questions, which suits rounds with many answers per branch.

## Limitations

- The guaranteed improvement is computed from sample means, so each cell is only as reliable as its row count, which `counts` reports.
- Atom significance treats words as evidence of disagreement. Two answers that say the same thing in different words look different.
- The loop has not been run on the Kaggle dataset it is configured for.

## Related samples

- [LoRA improvement loop](../../../../lora-improvement-loop/lora_improvement_loop/agent.py) - A coordinator with read-only specialists and a spec-verified training loop.
