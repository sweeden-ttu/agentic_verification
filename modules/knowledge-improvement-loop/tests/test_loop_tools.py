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

"""Tests for the loop tools."""

from __future__ import annotations

import json

from knowledge_loop import loop_tools as lt
import pytest


@pytest.fixture
def dirs(tmp_path, monkeypatch):
  work, base = tmp_path / "work", tmp_path / "kaggle"
  base.mkdir()
  rows = []
  for cand, easy, hard in (("base", 0.5, 0.5), ("risky", 0.9, 0.3), ("steady", 0.56, 0.55)):
    for sl, v in (("easy", easy), ("hard", hard)):
      rows += [{"cand": cand, "slice": sl, "score": v}] * 3
  (base / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
  monkeypatch.setenv("LOOP_WORKDIR", str(work))
  monkeypatch.setenv("KAGGLE_BASELINE_DIR", str(base))
  return work, base


def _j(s):
  return json.loads(s)


def test_missing_baseline_is_reported_not_raised(tmp_path, monkeypatch):
  """Off Kaggle the baseline path is absent and the tool says so."""
  monkeypatch.setenv("KAGGLE_BASELINE_DIR", str(tmp_path / "nope"))
  assert _j(lt.discover_baseline())["error_type"] == "BaselineMissing"


def test_robust_improvement_reads_the_baseline_table(dirs):
  """The tool returns the CFR policy and the average winner side by side."""
  r = _j(lt.robust_improvement("eval.jsonl", "cand", "slice", "score", "base"))
  assert r["best_on_average"] == "risky"
  assert max(r["robust_policy"], key=r["robust_policy"].get) == "steady"


@pytest.mark.parametrize("path", ["../x.jsonl", "/etc/passwd"])
def test_paths_cannot_leave_their_roots(dirs, path):
  """Traversal is refused for baseline tables and workdir files."""
  assert _j(lt.robust_improvement(path, "c", "s", "v", "b"))["error_type"] == "ValidationError"
  assert _j(lt.read_workdir_file(path))["error_type"] == "ValidationError"


def test_answers_votes_and_refinement_round_trip(dirs):
  """Recorded answers and votes feed refine_questions, which writes refined.json."""
  for _ in range(6):
    lt.record_answer("q1", "Which adapter?", "empirical", "use lora rank sixteen")
    lt.record_answer("q1", "Which adapter?", "skeptic", "use full finetune instead")
  for _ in range(5):
    lt.record_vote("q1", 0, True)
  lt.record_answer("q2", "Flat?", "empirical", "same words")
  lt.record_answer("q2", "Flat?", "skeptic", "same words")
  r = _j(lt.refine_questions(top_k=3))
  assert [q["id"] for q in r["kept"]] == ["q1"] and "q2" in r["dropped"]
  assert r["kept"][0]["best_answer"]["confidence"] > 0.9
  assert (dirs[0] / "refined.json").is_file()


def test_vote_on_unknown_answer_is_an_error(dirs):
  """Votes need an existing question and answer index."""
  assert _j(lt.record_vote("nope", 0, True))["error_type"] == "NotFound"
  lt.record_answer("q1", "Q", "empirical", "a")
  assert _j(lt.record_vote("q1", 3, True))["error_type"] == "ValidationError"


def test_adk_candidates_report_install_status(dirs):
  """Every catalog entry says whether it imports in this environment."""
  cands = _j(lt.adk_candidates())["candidates"]
  assert all("installed" in c for c in cands)
  assert any(c["name"] == "Workflow" and c["installed"] for c in cands)
