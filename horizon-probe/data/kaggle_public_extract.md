# Gemma 4 Developer Agent: forum and notebook evidence

Sources: 60 forum threads (`topic_*.txt`; the folder holds 60, not 61), `notebooks/index.csv`, and 13 notebooks (markdown and code; none had saved outputs; base64 blobs skipped). Notebooks are cited by file name. The full structured data is in `kaggle_public_extract.json`: 21 tools, 43 rules, 32 syntax items, 57 errors, 22 task/test items, 85 vocabulary terms, 38 disputed questions. Forum comments are truncated ("...") in the source files themselves.

## 1. Tools
- **Harness tools** (gemma4-baseline-agent-v2-compile-fixed, zero-to-first-valid-submission, what-moves-the-leaderboard):
  - `run_command(command)`: first 5,000 characters of output returned; default timeout 300 s; counts toward the tool budget.
  - `read_file(filepath, start_line, end_line)`: at most 150 lines or 10,000 characters per call.
  - `edit_file(filepath, old_string, new_string, allow_multiple)`: `old_string` must match byte for byte.
  - `write_file(filepath, content)`: repository paths only (not /tmp).
  - `get_status()` and `submit_patch()` are free; `submit_patch` runs `git add -N . && git diff HEAD`.
- **Graph tools:** `get_code_neighbors(node, edge_type, max_neighbors=50)` (`edge_type="CALLS"` returns nothing: only lowercase `calls` edges exist, and 0 async defs are nodes); `search_similar_code(query, k=10)` (existing node names only, full node source with no length cap, embeddings "near-collapsed": topic_744040, topic_744577, 119-of-129); `get_code_subgraph(nodes)`.
- **Composition:** `agent_tool: {config_path, skip_summarization}`; `run_skill_script(skill_name, file_path, args)` for skills.
- **Offline and validation APIs:** `validate_single_declared_model`, `validate_directory`, `compile_submission`, `build_submission_limits`, `discover_adapters`, `VllmConfig/VllmServer`, `Evaluator/EvalConfig`, the `swegemma eval` CLI, `kaggle competitions submit`.

## 2. Rules (allowed / banned / limits / scoring)
**Limits**
- One submission per day and two final picks (Rules 2.2a/2.2b, quoted in gemma-4-lb-58-tasks).
- 12 hours "inclusive of sandbox setup time, but excluding time for patch validation".
- About 120 hidden tasks from **private repositories**, ~60 public and ~60 private, run **sequentially** in one 12-hour run; host: "we don't support parallel calls".
- Context 32,768 tokens (prompt + reasoning + output); `max_output_tokens` 1 to 32,768 (default 16,384); `thinking_budget` default 4,096.
- Bundle under 3 GiB; allowed extensions .yaml, .yml, .md, .txt, .py, .json, .safetensors.
- The scorer reads only four `eval_config` keys (timeout_seconds, max_tool_calls, max_time_minutes, max_turns); host: "The default is no limit."
- Sandbox: "offline, 2 vCPU, 4 GB"; Docker `network_mode=none`; "The grader sandbox is air-gapped".
- GPU quota 30 h/week; L4x4 time charged at 2x.

**Allowed**
- Only `gemma-4-31b-it-qat-w4a16-ct` ("the only allowed model"), required "for every agent and subagent".
- PEFT LoRA adapters ("The rules explicitly permit PEFT LoRA adapters").
- Skills under `skills/<name>/SKILL.md`, up to 1,000 at 50 MiB each.
- External models "in accordance with their model license terms" (host reply, truncated).

**Banned**
- Private code sharing ("not permitted"); public code goes to the forum or notebooks. Under 13 not allowed; under 18 needs consent forms.
- Python entrypoints: "declarative YAML only ... No `agent.py`", compiled "against closed registries".
- `tools/system_instruction/http_options/safety_settings/response_schema` inside `generate_content_config` ("rejected ... by design").
- `..`, symlinks and absolute paths in `!include` ("blocked").
- Changing the chat template or the tool layer; editing tests, conftest.py or pytest.ini (the "Anti-Tampering Test & Config Reset").

**Scoring**
- Score is the share of tasks whose validation tests pass. HARNESS_README §8.2 says resolved means exit code 0; the swegemma code and the README re-issued on 2026-09-25 also require a JUnit report where "every required test node passed explicitly, and a skip does not count as a pass".
- Hitting 12 h "will error"; scoring unfinished tasks as 0 is planned but was "Not yet" live on Sep 28.
- Leaderboard truncates to 2 decimals; public split inferred as 58 tasks; ties go to whoever submitted first.

## 3. Syntax
- **agent.yaml:** fields `name, model, description, instruction, tools, skills, sub_agents, generate_content_config` (plus `include_contents`); exactly one root file (agent.yaml, agent.yml or root_agent.yaml); `!include` reads .md/.txt as raw text and .yaml recursively; `{problem_description?}` is an optional placeholder; sub-agent as tool: `- agent_tool: {config_path: sub_agents/x.yaml, skip_summarization: true}`.
- **sampling.yaml:** temperature, top_p, top_k, seed, max_output_tokens, `thinking_config{thinking_budget, include_thoughts}`. **Omit `thinking_level`** (it becomes `reasoning_effort`, which LiteLLM rejects). `include_thoughts:false` turns off `enable_thinking` entirely.
- **eval_config.yaml:** `evaluation: {timeout_seconds, max_tool_calls, max_time_minutes, max_turns}`; the host sample's 60/10/1/50 is a known pitfall.
- **SKILL.md:** YAML front matter (name, description) plus `scripts/`; run via `run_skill_script(skill_name=..., file_path="scripts/x.py", args=[...])`.
- **submission.zip:** agent.yaml at the zip root; optional `sub_agents/`, `skills/`, `adapters/<n>/(adapter_config.json + adapter_model.safetensors)`; a bytecode cache gets the bundle rejected by the compiler.
- **Tool calls:** Gemma template `<|tool_response>response:read_file{value:<|"|>...<|"|>}<tool_response|>`; tool error payload `{'status': 'error', 'error_type': 'FileEditError', ...}`; the parser hangs on `x:[a<|"|>]`.
- **Patches:** task diffs are plain `---/+++` with no `diff --git` header. The reset step `git checkout HEAD -- <files> 2>/dev/null || true` does nothing when a listed file is missing (bug). The graded pytest run adds `--junitxml -s` and `PYTHONNOUSERSITE=1`.

## 4. Errors (most frequent / important)
| Message (verbatim, shortened) | Stage | Cause / fix | Source |
|---|---|---|---|
| `Notebook Threw Exception` | grading | Oct 1 failures; host: "We had a GPU outage overnight. Resolved now" | 744800/802/805/807/813/861/924 |
| `Your notebook hit an unhandled error while rerunning your code` | grading | sample_submission as-is, an adapter submission, outage | 743140, 744331, 744807, 744800 |
| `...ContextWindowExceededError ... maximum context length is 32768 tokens. However, you requested 2048/16384 output tokens and your prompt contains at least 30721/16385 input tokens` | model | prompt+output > 32,768; overflow discards the patch | prvsiyan, 16384-cap nb, 744577, 744692 |
| `Failed to replace: old_string not found. Ensure you're not escaping content incorrectly ...` | runtime | double-JSON tool output; fixed in the 09-30 wheelhouse | 744272, 745003 |
| `ValueError: Tool 'search_similar_code' not found. Available tools: ...` | runtime | undeclared or hallucinated tool ends the task (score 0) | 745052, 745028 |
| `ValueError: Gemma4ForConditionalGeneration does not support LoRA yet` | model | stock vLLM 0.19.1; use the patched wheelhouse | 744070 |
| `No LoRA weights found for module ...self_decoder.decoder_layers.N..., skipping.` | model | adapter silently zeroed; fixed in v23 | 743508 |
| `Running: 0 reqs, Waiting: 1 reqs` | model | LoRA shrinks the KV cache to 7,600 tokens; params now dynamic | 744331, 744794 |
| `Failed to apply test_patch: ... Reversed (or previously applied) patch detected` | grading | test-file reset is a no-op ("Addressing now") | 744825, prvsiyan |
| `ModuleNotFoundError: No module named 'typing_inspection'` | grading | missing wheels and a stale wheel cache; hidden set validates 100% | 744029, 744370, 744259 |
| `'>' not supported between instances of 'int' and 'str'` | runtime | read_file line ranges; fixed | 744678 |
| `Your notebook has requested more CPU, GPU or TPU resources than are available.` | grading | platform; resolved and reruns promised | 743683 |
| `Did not find output file 'submission.zip' in the submitted Notebook.` | packaging | submitted before saving a version (WAI) | 743683 |
| `Agent exceeded session timeout (5 min)` / `... tool call budget (45 calls)` / `BudgetExceeded` | runtime | budget caps | prvsiyan, cpu-audit-tables |
| `Disallowed file extension` / `Archive exceeds 3 GiB limit` / bytecode cache rejected | packaging/compile | bundle checks | getting-started, prvsiyan |
| `unsupported reasoning_effort` (thinking_level) | model | LiteLLM rejects it | prvsiyan, getting-started |

Other errors in the JSON: gold-patch failures in public tasks (anyio disabled, the CPython 3.13 message change, pinned pytest, VERIFY_X509_STRICT); tasks that pass with an empty patch; DNS-blocked Requests tests; src-layout imports; the base_commit mismatch; the parser infinite loop; dropped thoughts; seed and thinking_budget not forwarded; compaction losing tool results; the 12-hour overrun error; queue, quota and batch-session limits.

## 5. Tasks and tests
- **Public data:** 129 practice tasks (fastapi 67, rich 48, requests 13, httpx 1), Python bug fixes and feature requests. 64% of fixes change ≤20 lines (median 12); 71% touch one file; 76% of issues name neither the file nor the function; 16% touch async code.
- **Grading:** the patch is applied in a fresh Container B with the hidden tests and must pass. Host: hidden tasks are "more rigorously filtered" and gold patches score 100%.
- **Public-set audits:** 119/129 sound (117 under a surrogate), range 2/129 (empty patch) to 125/129; a second audit finds 114/129 sound; 18 of 40 gold patches pass in the public sandbox (3 of 23 FastAPI); verifier environments 43/3/40 → 107/2/105 gold/empty/valid.
- **Scores reported:** leaderboard wall ~0.12, leader 0.13 (Sep 25); 0.08 for several bundles; 4/58 = 6.90% against a predicted 17/58; CV and LB poorly correlated (0.23 → 0.12, 0.24 → 0.10); a one-task lead near the medal cut stays ahead on private only 0.47-0.49 of the time.
- **Time budget:** about 4.5 min per task at 120 tasks; `max_time_minutes` ≤ 4.94-5.98; measured runs took 14.5 h, and one hit a "timeout after 14 hours".

## 6. Vocabulary (case-insensitive, `\b` word boundary, all 74 files; code included, base64 removed)
| Sense | Term: total (forum/notebooks) |
|---|---|
| legal | allowed 16 (6/10), allow 17, permitted 11 (2/9), permit 2, permissible 1, valid 24 (1/21, csv 2), supported 23, accepted 8, WAI 1 |
| banned | banned 1, not allowed 1, not permitted 1, disallowed 4, forbidden 7 (all prvsiyan), blocked 2, rejected 5, reject 9, refused 24, refuses/refusal 19, invalid 18, must not 8, never 120, unsupported 5, off the table 1, against their ToS 1, breaking the rules 1, air-gapped 1, network_mode=none 1, closed registries 2, Anti-Tampering 1, reward hacking 1 |
| ambiguous | clarify* 9 (all forum), confirm* 28, unclear 9, ambiguous 6, not sure 2, unresolved 37, bug(s) 26, feature 7, acceptable 2, eligibility 2 |
| other | only 340, required 160, budget 152, limit 151, sandbox 115, must 98, validate* 82, cap 76, offline 41, legal 1 ("legal resident") |
| zero hits | prohibited, prohibit, forbid, illegal, ban, disqualified/disqualify, terms of service, denied, restricted, violate, cheat, eligible, unsure |

In the forum, explicit allow/ban words are rare; rules mostly appear as questions ("Could you clarify", "would that be allowed, or would it count as breaking the rules?"). The hard prohibitions ("rejected ... by design", "blocked", "forbidden", "refuses") come from participants' notebooks reading the harness.

## 7. Disputed / unsure (38 in JSON; key ones)
1. **Distillation from proprietary APIs** (742807): the host's conditional reply ("in accordance with their model license terms ...") is truncated; one participant asked "So models like Claude and GPT are off the table?"; others warned it is "against their ToS". Unclear.
2. **Winner release of weights/adapters (Sections 2.5, 2.8):** unanswered.
3. **Training on self-generated or other open-weight outputs** (743964): unanswered.
4. **Agent adding a missing package in its patch** ("breaking the rules?"): unanswered; participant prompts forbid installing packages.
5. **Which base models:** "the only allowed model" vs a notebook implying "bf16 31B/27B fit". Resolved by the quoted rules: 31B QAT only.
6. **LoRA on the scorer:** permitted, but three bugs; host says fixed (v23/v25, dynamic params); one adapter submission still errored. Partially resolved.
7. **12-hour overrun** (error vs unfinished tasks scored 0): planned, "Not yet". Whether queue time counts: unanswered.
8. **eval_config defaults:** host "default is no limit" vs harness code 300 s / 60 min / 500 turns. Conflicting.
9. **Compaction threshold:** 32,768 vs 14,336, interval 5 vs 15. Unresolved ("I will look into it").
10. **Double-encoded tool output, bug or "feature":** a bug, fixed. **Dropped thoughts:** "unambiguously a bug", fixed.
11. **`include_thoughts` semantics; thinking_budget/seed forwarding:** forwarding patched; semantics unresolved.
12. **Are tasks solvable** (gold-patch failures): yes for the hidden set (host); the public set is still broken.
13. **README rule (exit 0) vs code (JUnit + required nodes):** resolved by the 2026-09-25 README re-issue.
14. **`.py` skill scripts in the sandbox under "no agent.py":** unclear. **Callbacks in YAML:** unanswered.
15. **Undeclared tools ending tasks:** pending. **search_similar_code cap:** host committed.
16. **Test-reset no-op, embeddings collapse, src-layout imports, snapshot/base_commit mismatch, chat-template update:** pending or unanswered.
17. **Reruns or allowance restores after infra failures:** unresolved.
18. **CPU-only submission notebooks; Writeup-linked code sharing; prize payout to Russian residents; TPU support:** unanswered.
19. **`!include ../`:** "blocked" per one notebook, whose own bundle nevertheless uses `../prompts/analyzer.md`.
