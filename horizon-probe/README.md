# horizon-probe

Boundary analysis of where models diverge when asked the very same question, with "legal" and "banned" as two separate axes.

Two probes, one analysis:

1. **Legal / banned item bank** (`data/items.json`, 38 statements). Built from the rules, syntax, errors and disputed questions in the Gemma 4 Developer Agent forum threads and notebooks (`data/kaggle_public_extract.json`, extracted from `kaggle_public/`). Every model gets the identical prompt with four options:
   - A) legal, not banned
   - B) legal, but banned
   - C) illegal, and banned
   - D) illegal, but not banned

   Statements whose rule is explicit carry an anchor (`a` or `b`); disputed ones carry none.
2. **Recursive root question**, asked verbatim, typo included: "What are the boundary tokens we find meaningul surprise in the divergence of its answers?"
   - Level 0 shows the models the statements they disagreed on most.
   - Level k shows them the boundary tokens and one anonymous excerpt per model from level k-1, then asks the same question again.
   - Recursion stops at a fixed point (the boundary-token set overlaps the previous level's by Jaccard ≥ 0.5), when no boundary tokens remain, or at the depth limit.
   - Models appear as M1..Mn, reshuffled each level, so no model sees names.

## Bayesian decomposition

For every unit u (an answer letter for the item bank; a word or word pair for free text) and every model m, the model's rate gets a Jeffreys Dirichlet/Beta posterior (α = 0.5).

```
H[ p̄(u) ]      =  mean_m H[ p_m(u) ]   +   I(u ; model)
total surprise    within-model noise       between-model divergence  (meaningful surprise)
```

- **Meaningful share** = I / H, reported per item, per token and per recursion level.
- **Bayes factor**: log BF compares "each model has its own rate" against "one shared rate", using Dirichlet-multinomial marginal likelihoods. A boundary item or token needs I ≥ threshold *and* log BF ≥ 1.0 (`config/run.json`).
- **Surprise per model**: KL(p_m ‖ mean of the other models), in bits. That is how much a model's answers surprise everyone else. Each model's signature tokens are ranked by it.
- **Axis split**: the item bank is decomposed twice more, on P(legal) = A + B and on P(banned) = B + C. Comparing the two divergences says which axis a disagreement lives on.

## Deep belief network

For each recursion level, two stacked Bernoulli RBMs (32 then 8 hidden units) are trained on the answer × token presence matrix. Each top-layer unit is scored by the mutual information between its activation and model identity, and is listed with its top back-projected tokens.

With a few answers per model this is exploratory. The report always lists it next to the Bayesian boundary tokens and shows their overlap.

## Prolog

`prolog/horizon.pl` runs on the facts written to `runs/<run>/facts.pl`:

| Predicate | Meaning |
|---|---|
| `side/4`, `quadrant/3` | Each model's position on the legal axis, the banned axis and A–D |
| `boundary_item/1`, `divergence_axis/2`, `split/4` | Where models disagree and on which axis |
| `universal/2`, `universal_side/3` | What every model agrees on: the shared definition |
| `tier_pivot/3` | Items where every larger model sits on one side and every smaller model on the other |
| `conflation_rate/3` | How often a model reads an explicitly banned-but-legal rule as illegal (B answered as C) |
| `anchor_accuracy/3` | Agreement with the explicit rules |
| `horizon/5` | Per model and item kind, how many items it puts on the legal or banned side |
| `persistent_token/1`, `fixed_point_token/1`, `axis_token/3` | Recursion tokens that survive levels, and the ones that carry legal, banned or ambiguous vocabulary (`data/lexicon.json`) |

## Setup

```bash
scripts/install_prolog.sh                       # SWI-Prolog into .tools/env (repo-local, via micromamba)
WITH_LLAMA_CPP=1 scripts/install_prolog.sh      # also llama.cpp, for running Gemma GGUF locally
python3 -m pip install -r requirements.txt
scripts/fetch_gemma_gguf.sh E2B E4B             # optional: Gemma 4 QAT q4_0 GGUF into models/
export ANTHROPIC_API_KEY=...                    # for opus and sonnet
```

Edit `config/models.json`:
- **Anthropic models (Opus, Sonnet):** use the Messages API. There are no logprobs, so they are sampled k times at temperature 1.
- **OpenAI-compatible servers (vLLM, llama.cpp, LM Studio):** use exact first-token logprobs over A–D when `logprobs` is true. Gemma 4 thinking is turned off through `chat_template_kwargs`.
- **`launch` block:** starts that server before the model's calls and stops it after, so models that don't fit in memory together run one at a time.
- **`rank`:** orders models by size for `tier_pivot`.

## Run

```bash
python3 -m horizon all --run runs/full                    # item bank, recursion, DBN, Prolog, report
python3 -m horizon mcq --run runs/full --models opus,sonnet
python3 -m horizon recurse --run runs/full --depth 4 --k 8
python3 -m horizon analyze --run runs/full                # recompute everything from saved answers
```

Runs resume: answers already in `runs/<run>/**.jsonl` are not requested again.

## Outputs

All under `runs/<run>/`:
- `meta.json`: models, config and item file used.
- `mcq/<model>.jsonl`, `recurse/level_k/<model>.jsonl`: raw answers. Each level also has `prompt.txt`, `labels.json`, `analysis.json` and `dbn.json`.
- `mcq_analysis.json`, `recurse_summary.json`, `dbn_summary.json`.
- `facts.pl`, `prolog_results.json`.
- `report.md`, `report.json`.

`runs/smoke-gemma-e4b-e2b` is a CPU smoke run of Gemma 4 E4B and E2B only (k = 4, depth 3). It checks the pipeline end to end. With two small models it is not a result about Opus or Sonnet.
