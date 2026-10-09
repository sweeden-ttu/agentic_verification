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

"""Tools for the knowledge-improvement loop.

Every tool returns a JSON string with ``{"status": "ok", ...}`` or
``{"status": "error", "error_type": ..., "error_message": ...}``. Loop state
lives in ``LOOP_WORKDIR`` (default ``./work``): ``nodes.json`` holds the
question graph with every branch's answers and verifier votes, and
``improvements.json`` holds what a human approved. Baseline paths are relative
to ``KAGGLE_BASELINE_DIR``. Traversal outside either root is rejected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import adk_catalog
from . import baseline
from . import refine

MAX_CHARS = 10_000


def workdir() -> Path:
  return Path(os.environ.get("LOOP_WORKDIR", "work")).resolve()


def _ok(**fields: Any) -> str:
  return json.dumps({"status": "ok", **fields}, default=str)


def _error(error_type: str, message: str, **details: Any) -> str:
  payload: dict[str, Any] = {"status": "error", "error_type": error_type, "error_message": message}
  if details:
    payload["details"] = details
  return json.dumps(payload)


def _inside(root: Path, rel: str) -> Path:
  if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
    raise ValueError("path must be relative, without '..'")
  root = root.resolve()
  p = (root / rel).resolve()
  if p != root and root not in p.parents:
    raise ValueError("path leaves its root")
  return p


def _load_nodes() -> list[dict[str, Any]]:
  p = workdir() / "nodes.json"
  return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


def _save_nodes(nodes: list[dict[str, Any]]) -> None:
  workdir().mkdir(parents=True, exist_ok=True)
  (workdir() / "nodes.json").write_text(json.dumps(nodes, indent=1), encoding="utf-8")


def adk_candidates() -> str:
  """Lists ADK components that could join the loop, checked against the install.

  Returns:
    A JSON string with each candidate's import path, installed flag and use.
  """
  return _ok(candidates=adk_catalog.catalog())


def discover_baseline() -> str:
  """Lists the Kaggle baseline's files and the columns of its tables (read-only).

  Returns:
    A JSON string with the baseline root and its files.
  """
  root = baseline.baseline_root()
  if not root.is_dir():
    return _error("BaselineMissing", f"{root} does not exist here; run on Kaggle or set KAGGLE_BASELINE_DIR.")
  return _ok(root=str(root), files=baseline.discover(root))


def robust_improvement(table_path: str, candidate_col: str, slice_col: str, score_col: str, baseline_name: str) -> str:
  """Solves for the candidate mix with the best guaranteed improvement (CFR+).

  Args:
    table_path: Table under the baseline root (CSV or JSON Lines).
    candidate_col: Column naming the candidate (prompt, adapter, agent).
    slice_col: Column naming the evaluation slice.
    score_col: Numeric score column; higher is better.
    baseline_name: The candidate value that is the current baseline.

  Returns:
    A JSON string with the payoff matrix, the robust policy, the guaranteed
    improvement, and the best-on-average candidate for comparison.
  """
  try:
    path = _inside(baseline.baseline_root(), table_path)
    rows = baseline.read_rows(path)
    result = baseline.robust_choice(rows, candidate_col=candidate_col, slice_col=slice_col,
                                    score_col=score_col, baseline=baseline_name)
  except FileNotFoundError:
    return _error("FileNotFound", f"No such table: {table_path}")
  except (ValueError, KeyError) as e:
    return _error("ValidationError", str(e))
  return _ok(**result)


def read_workdir_file(filepath: str) -> str:
  """Reads a file in the loop's working directory (read-only, 10,000 chars).

  Args:
    filepath: Path relative to the working directory.

  Returns:
    A JSON string with the content.
  """
  try:
    p = _inside(workdir(), filepath)
  except ValueError as e:
    return _error("ValidationError", str(e))
  if not p.is_file():
    return _error("FileNotFound", f"No such file: {filepath}")
  return _ok(content=p.read_text(encoding="utf-8", errors="replace")[:MAX_CHARS])


def record_answer(question_id: str, question: str, branch: str, answer: str, parent_id: str = "") -> str:
  """Adds one branch's answer to a question node, creating the node if new.

  Args:
    question_id: Short id such as ``q1`` or ``q1.2``.
    question: The question text.
    branch: Which researcher answered, such as ``empirical`` or ``skeptic``.
    answer: The answer, with its evidence.
    parent_id: Id of the question this one follows up, or empty for a root.

  Returns:
    A JSON string with the node's answer count.
  """
  if not question_id.strip() or not answer.strip() or not branch.strip():
    return _error("ValidationError", "question_id, branch and answer must not be empty")
  nodes = _load_nodes()
  node = next((n for n in nodes if n["id"] == question_id), None)
  if node is None:
    node = {"id": question_id, "parent": parent_id or None, "question": question, "answers": []}
    nodes.append(node)
  node["answers"].append({"branch": branch, "text": answer, "votes": []})
  _save_nodes(nodes)
  return _ok(question_id=question_id, answers=len(node["answers"]))


def record_vote(question_id: str, answer_index: int, supported: bool) -> str:
  """Records a verifier's judgement of whether an answer's evidence holds.

  Args:
    question_id: The question node.
    answer_index: Index of the answer within the node, from 0.
    supported: True if the evidence supports the answer.

  Returns:
    A JSON string with the answer's vote count.
  """
  nodes = _load_nodes()
  node = next((n for n in nodes if n["id"] == question_id), None)
  if node is None:
    return _error("NotFound", f"no question {question_id}")
  if not 0 <= answer_index < len(node["answers"]):
    return _error("ValidationError", f"answer_index must be in 0..{len(node['answers']) - 1}")
  node["answers"][answer_index]["votes"].append(bool(supported))
  _save_nodes(nodes)
  return _ok(votes=len(node["answers"][answer_index]["votes"]))


def refine_questions(top_k: int = 5, followups_per_question: int = 3) -> str:
  """Keeps the questions with the most significant atoms and surest answers.

  Args:
    top_k: How many questions to keep.
    followups_per_question: How many follow-ups to propose for each.

  Returns:
    A JSON string with kept questions, their significant atoms, best answer,
    confidence and follow-ups, and the ids that were dropped.
  """
  nodes = _load_nodes()
  if not nodes:
    return _error("Empty", "no questions recorded yet")
  result = refine.refine(nodes, top_k=top_k, followups_per_question=followups_per_question)
  (workdir() / "refined.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
  return _ok(**result)


def promote_improvement(title: str, evidence: str, expected_gain: float) -> str:
  """Records an improvement for the team to adopt. A human must confirm it.

  Args:
    title: What to change, in one line.
    evidence: The numbers and sources behind it.
    expected_gain: The guaranteed improvement on the score from robust_improvement.

  Returns:
    A JSON string with the number of promoted improvements.
  """
  if not title.strip() or not evidence.strip():
    return _error("ValidationError", "title and evidence must not be empty")
  path = workdir() / "improvements.json"
  items = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
  items.append({"title": title, "evidence": evidence, "expected_gain": expected_gain})
  workdir().mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(items, indent=1), encoding="utf-8")
  return _ok(promoted=len(items))
