# Boundary analysis

Root question (verbatim): "What are the boundary tokens we find meaningul surprise in the divergence of its answers?"

Every quantity below splits the same total surprise two ways: within-model noise (the average entropy of each model's own answers) and between-model divergence (the mutual information between the answer and which model gave it). Only the second part is meaningful surprise; boundary items and boundary tokens are where it is large and the Bayes factor says the models really differ.

## Legal and banned: the same question to every model

38 statements. Total surprise 64.1 bits = within-model noise 55.9 + between-model divergence 8.2 (13% meaningful). Boundary items: 24.

| Model | Rank | A legal, not banned | B legal, but banned | C illegal, banned | D illegal, not banned | Mean P(legal) | Mean P(banned) | Banned read as illegal | Anchor accuracy |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b | 2 | 31 | 2 | 5 | 0 | 0.73 | 0.39 | 25% | 37% |
| gemma4-e2b | 1 | 1 | 36 | 1 | 0 | 0.84 | 0.75 | 0% | 68% |

### Statements with the most between-model divergence

| Item | Kind | Divergence (bits) | log BF | Axis | gemma4-e4b | gemma4-e2b | Statement |
|---|---|---|---|---|---|---|---|
| i16 | rule_allowed | 0.40 | 3.3 | banned | A | B | A team publishes its agent's code in a public Kaggle notebook. |
| i36 | disputed | 0.39 | 2.6 | legal | A | C | Kaggle pays a prize to a resident of Russia who is not on any sanctions list. |
| i07 | rule_banned | 0.39 | 4.1 | banned | A | B | A team adds a custom tool that post-processes tool output before the model sees it. |
| i02 | rule_banned | 0.39 | 4.4 | legal | C | B | A team submits a Python agent.py entrypoint instead of a declarative agent.yaml. |
| i34 | disputed | 0.35 | 3.6 | banned | A | B | A team trains its LoRA adapter on Kaggle's 4xL4 notebooks instead of its own GPUs. |
| i17 | rule_allowed | 0.30 | 2.6 | banned | A | B | A team uses a sub-agent as a tool through agent_tool with config_path. |
| i10 | rule_banned | 0.29 | 2.2 | banned | A | B | A team makes two submissions on the same day. |
| i01 | rule_banned | 0.28 | 2.8 | banned | A | B | A team's agent.yaml declares a sub-agent that runs on gemma-4-E4B instead of gemma-4-31b-it-qat-w4a16-ct. |
| i24 | error | 0.27 | 2.2 | legal | C | B | The agent calls a tool named search_similar_code that is not declared in agent.yaml. |
| i08 | rule_banned | 0.27 | 2.4 | banned | A | B | The agent runs pip install inside the grading sandbox to fetch a package. |
| i31 | disputed | 0.27 | 2.1 | banned | A | B | A team uses before_model_callbacks in agent.yaml. |
| i30 | disputed | 0.25 | 1.9 | banned | A | B | A skill ships a .py script that the agent runs with run_skill_script inside the sandbox. |

### Universal answers (every model agrees)

i15=A, i22=B

## Recursion of the root question

Stopped: depth reached.

| Level | Vocabulary | Total bits | Noise bits | Divergence bits | Meaningful share | Jaccard to previous | Top boundary tokens |
|---|---|---|---|---|---|---|---|
| 0 | 110 | 99.7 | 90.2 | 9.5 | 10% |  | implementation, international, international sanctions, legality, model, technical |
| 1 | 102 | 90.2 | 77.4 | 12.9 | 14% | 0.00 | chosen, chosen option, instances, leading, option, option seems, reasoning leading, two |
| 2 | 100 | 88.6 | 80.7 | 7.8 | 9% | 0.10 | appear, different frequency, chosen, chosen option, different, identifies, information, information provided |

### Boundary tokens at level 2

| Token | Divergence (bits) | log BF | gemma4-e4b | gemma4-e2b | Sense |
|---|---|---|---|---|---|
| appear | 0.53 | 4.2 | 10% | 90% |  |
| different frequency | 0.53 | 4.2 | 10% | 90% |  |
| chosen | 0.30 | 2.1 | 90% | 30% |  |
| chosen option | 0.30 | 2.1 | 90% | 30% |  |
| different | 0.30 | 2.1 | 30% | 90% |  |
| identifies | 0.30 | 2.1 | 30% | 90% |  |
| information | 0.30 | 2.1 | 10% | 70% |  |
| information provided | 0.30 | 2.1 | 10% | 70% |  |
| m2 identifies | 0.30 | 2.1 | 10% | 70% |  |
| option | 0.30 | 2.1 | 90% | 30% |  |
| option instances | 0.30 | 2.1 | 90% | 30% |  |
| significantly different | 0.30 | 2.1 | 30% | 90% |  |
| signify | 0.30 | 2.1 | 30% | 90% |  |

Signature tokens (used more by this model than by any other, ranked by the surprise they cause the others):

- gemma4-e4b: chosen, option instances, option, chosen option, usage, response, markers, use
- gemma4-e2b: different frequency, appear, m2 identifies, information, information provided, different, significantly different, identifies

### Fixed-point tokens (boundary at the last two levels)

chosen, chosen option, option

### Tokens that stay on the boundary from one level to the next

chosen, chosen option, option

### Boundary tokens on the legal, banned or ambiguous axis

0 international sanctions legal, 0 legality legal

## Deep belief network (stacked RBMs over token presence)

Exploratory: with a few answers per model, the RBM units are unstable; read them next to the Bayesian boundary tokens, not instead of them.

- Level 0: layers [32, 8], most model-specific unit carries 0.19 of 1.00 bits; its top terms: international sanctions, international, implementation, legality, technical, significant, sanctions list, technical implementation; overlap with Bayesian boundary: implementation, international, international sanctions, legality, technical
- Level 1: layers [32, 8], most model-specific unit carries 1.00 of 1.00 bits; its top terms: two models, option seems, option, two, leading, instances, chosen option, chosen; overlap with Bayesian boundary: chosen, chosen option, instances, leading, option, option seems, two, two models
- Level 2: layers [32, 8], most model-specific unit carries 0.19 of 1.00 bits; its top terms: option instances, option, chosen, chosen option, chosen chosen, leading, two two, leading two; overlap with Bayesian boundary: chosen, chosen option, m2 identifies, option, option instances
