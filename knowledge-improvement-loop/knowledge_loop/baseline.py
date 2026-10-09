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

"""Reads a Kaggle evaluation baseline and picks the robust improvement.

The baseline is any table of per-row scores, such as an Evidently report
export: one row per evaluated example, with a column naming the candidate
(baseline prompt, adapter, agent variant), a column naming the slice (task
type, repository, descriptor bucket) and a numeric score column. CSV and
JSON Lines are read with the standard library.

``robust_choice`` builds the matrix of mean improvement over the baseline
candidate for every candidate and slice, then solves it with CFR+ as a game
against an adversary who picks the slice. The result is the mix of candidates
with the best guaranteed improvement, which can differ from the candidate with
the best average.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from . import cfr

DEFAULT_ROOT = "/kaggle/input/datasets/scottweeden/gemma4-evidently-examples"
TABLE_SUFFIXES = (".csv", ".jsonl", ".json")


def baseline_root() -> Path:
  return Path(os.environ.get("KAGGLE_BASELINE_DIR", DEFAULT_ROOT))


def discover(root: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
  """Lists files under the baseline root, with columns for readable tables."""
  root = root or baseline_root()
  out = []
  for p in sorted(root.rglob("*"))[:limit]:
    if not p.is_file():
      continue
    entry: dict[str, Any] = {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
    if p.suffix in TABLE_SUFFIXES:
      try:
        rows = read_rows(p, limit=5)
        entry["columns"] = sorted({k for r in rows for k in r})
      except (ValueError, OSError) as e:
        entry["error"] = str(e)
    out.append(entry)
  return out


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
  if path.suffix == ".csv":
    with open(path, newline="", encoding="utf-8") as f:
      rows = list(csv.DictReader(f))
  elif path.suffix == ".jsonl":
    with open(path, encoding="utf-8") as f:
      rows = [json.loads(line) for line in f if line.strip()]
  else:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", [])
  return rows[:limit] if limit else rows


def score_matrix(rows, *, candidate_col: str, slice_col: str, score_col: str, baseline: str) -> dict[str, Any]:
  """Mean score per candidate and slice, minus the baseline's mean on that slice."""
  sums: dict[tuple[str, str], list[float]] = {}
  for r in rows:
    try:
      v = float(r[score_col])
    except (KeyError, TypeError, ValueError):
      continue
    sums.setdefault((str(r[candidate_col]), str(r[slice_col])), []).append(v)
  cands = sorted({c for c, _ in sums})
  slices = sorted({s for _, s in sums})
  if baseline not in cands:
    raise ValueError(f"baseline candidate {baseline!r} not in {cands}")
  missing = [(c, s) for c in cands for s in slices if (c, s) not in sums]
  if missing:
    raise ValueError(f"no rows for candidate/slice pairs {missing[:5]}")
  mean = {k: sum(v) / len(v) for k, v in sums.items()}
  others = [c for c in cands if c != baseline]
  payoff = [[mean[(c, s)] - mean[(baseline, s)] for s in slices] for c in [baseline] + others]
  counts = {f"{c}|{s}": len(v) for (c, s), v in sums.items()}
  return {"candidates": [baseline] + others, "slices": slices, "payoff": payoff, "counts": counts}


def robust_choice(rows, *, candidate_col: str, slice_col: str, score_col: str, baseline: str,
                  iterations: int = 3000) -> dict[str, Any]:
  """CFR+ policy over candidates against the worst slice, plus the plain averages."""
  m = score_matrix(rows, candidate_col=candidate_col, slice_col=slice_col, score_col=score_col, baseline=baseline)
  solved = cfr.solve_matrix(m["payoff"], rows=m["candidates"], cols=m["slices"], iterations=iterations)
  averages = {c: sum(r) / len(r) for c, r in zip(m["candidates"], m["payoff"])}
  worst = {c: min(r) for c, r in zip(m["candidates"], m["payoff"])}
  return {
      **m,
      "robust_policy": solved["row_policy"],
      "hardest_slices": solved["col_policy"],
      "guaranteed_improvement": solved["value"],
      "exploitability": solved["exploitability"],
      "mean_improvement": averages,
      "worst_slice_improvement": worst,
      "best_on_average": max(averages, key=averages.get),
  }
