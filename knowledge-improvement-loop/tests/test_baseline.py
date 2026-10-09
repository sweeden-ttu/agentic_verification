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

"""Tests for the baseline adapter."""

from __future__ import annotations

import csv
import json

from knowledge_loop import baseline
import pytest


def _rows():
  # risky wins big on 'easy' and loses on 'hard'; steady is a little better everywhere.
  out = []
  for cand, easy, hard in (("base", 0.5, 0.5), ("risky", 0.9, 0.3), ("steady", 0.56, 0.55)):
    for sl, v in (("easy", easy), ("hard", hard)):
      out += [{"cand": cand, "slice": sl, "score": v}] * 3
  return out


def test_robust_choice_prefers_the_candidate_that_never_loses():
  """CFR picks 'steady'; the plain average would pick 'risky'."""
  r = baseline.robust_choice(_rows(), candidate_col="cand", slice_col="slice", score_col="score", baseline="base")
  assert r["best_on_average"] == "risky"
  assert max(r["robust_policy"], key=r["robust_policy"].get) == "steady"
  assert r["guaranteed_improvement"] > 0.04 and r["exploitability"] < 0.01


def test_payoff_is_relative_to_the_baseline():
  """The baseline row of the payoff matrix is all zeros."""
  m = baseline.score_matrix(_rows(), candidate_col="cand", slice_col="slice", score_col="score", baseline="base")
  assert m["candidates"][0] == "base" and m["payoff"][0] == [0.0, 0.0]


def test_missing_cells_and_unknown_baseline_are_errors():
  """Every candidate needs rows on every slice, and the baseline must exist."""
  rows = [r for r in _rows() if not (r["cand"] == "risky" and r["slice"] == "hard")]
  with pytest.raises(ValueError):
    baseline.score_matrix(rows, candidate_col="cand", slice_col="slice", score_col="score", baseline="base")
  with pytest.raises(ValueError):
    baseline.score_matrix(_rows(), candidate_col="cand", slice_col="slice", score_col="score", baseline="nope")


def test_discover_lists_tables_with_columns(tmp_path, monkeypatch):
  """CSV and JSONL files are listed with their column names."""
  with open(tmp_path / "eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cand", "slice", "score"])
    w.writeheader()
    w.writerows(_rows())
  (tmp_path / "more.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()))
  monkeypatch.setenv("KAGGLE_BASELINE_DIR", str(tmp_path))
  found = {e["path"]: e for e in baseline.discover()}
  assert found["eval.csv"]["columns"] == ["cand", "score", "slice"]
  assert "columns" in found["more.jsonl"]
  rows = baseline.read_rows(tmp_path / "eval.csv")
  assert baseline.robust_choice(rows, candidate_col="cand", slice_col="slice", score_col="score", baseline="base")
