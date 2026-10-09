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

"""Tests for the deep-belief-network refinement."""

from __future__ import annotations

from knowledge_loop import refine


def _node(qid, a_texts, b_texts, votes=None):
  answers = [{"branch": "A", "text": t} for t in a_texts] + [{"branch": "B", "text": t} for t in b_texts]
  if votes:
    answers[0]["votes"] = votes
  return {"id": qid, "question": f"question {qid}", "answers": answers}


def test_atoms_drop_stopwords_and_keep_negations():
  """Atoms are content words and pairs; 'not' survives as a negation."""
  out = refine.atoms("The adapter is not loaded")
  assert {"adapter", "not", "loaded", "not loaded"} <= out and "the" not in out


def test_atom_used_by_one_branch_only_is_significant():
  """An atom every A answer uses and no B answer uses splits the branches."""
  q = refine.analyze_question(_node("q", ["use lora rank"] * 6, ["use full finetune"] * 6))
  names = {a["atom"] for a in q["significant_atoms"]}
  assert "lora" in names and "use" not in names


def test_branches_that_agree_have_no_significant_atoms():
  """Identical answers carry no between-branch information."""
  q = refine.analyze_question(_node("q", ["same answer here"] * 5, ["same answer here"] * 5))
  assert q["significant_atoms"] == [] and q["question_value"] == 0


def test_confidence_is_the_beta_posterior_mean():
  """Nine of ten agreeing votes give (0.5 + 9) / 11."""
  assert abs(refine.confidence([True] * 9 + [False]) - 9.5 / 11) < 1e-12
  assert refine.confidence([]) == 0.5


def test_value_ranks_split_questions_with_confident_answers_first():
  """Same split, more confident best answer, higher question value."""
  split = (["use lora rank"] * 6, ["use full finetune"] * 6)
  sure = refine.analyze_question(_node("sure", *split, votes=[True] * 10))
  unsure = refine.analyze_question(_node("unsure", *split, votes=[True, False]))
  assert sure["question_value"] > unsure["question_value"]


def test_followups_go_where_disagreement_is_unsettled():
  """A settled answer leaves less information to gain from a follow-up."""
  split = (["use lora rank"] * 6, ["use full finetune"] * 6)
  sure = refine.analyze_question(_node("sure", *split, votes=[True] * 30))
  open_ = refine.analyze_question(_node("open", *split))
  assert open_["followups"][0]["value_of_information"] > sure["followups"][0]["value_of_information"]
  assert "'" in open_["followups"][0]["question"]


def test_refine_keeps_top_k_and_drops_flat_questions():
  """Questions without significant atoms are dropped; at most k are kept."""
  nodes = [
      _node("split1", ["use lora rank"] * 6, ["use full finetune"] * 6, votes=[True] * 5),
      _node("split2", ["cache prompts"] * 6, ["stream tokens"] * 6),
      _node("flat", ["same"] * 4, ["same"] * 4),
  ]
  out = refine.refine(nodes, top_k=1, followups_per_question=2)
  assert [q["id"] for q in out["kept"]] == ["split1"]
  assert set(out["dropped"]) == {"split2", "flat"}
  assert len(out["kept"][0]["followups"]) <= 2
