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

"""Refines a deep belief network of questions to its most significant atoms.

A *question node* holds answers from several independent branches (sub-agents
or models) and, optionally, verifier votes on each answer. An *atom* is the
smallest unit of an answer: a content word or an adjacent word pair, extracted
the same way as ``horizon-probe/horizon/terms.py``.

For each question the module measures:

- **Atom significance**: for each atom, the counts of answers that do and do
  not contain it, per branch, are decomposed as in ``horizon-probe``: total
  surprise = within-branch noise + I(atom; branch). An atom is significant when
  I >= ``mi_min_bits`` and ln BF(branch-specific vs shared rate) >= ``log_bf_min``.
  Significant atoms are where the branches really disagree.
- **Answer confidence**: the posterior mean of a Beta(0.5 + agree, 0.5 +
  disagree) over verifier votes. With no votes the confidence is 0.5.
- **Question value**: total MI of significant atoms times the best answer's
  confidence, so a question ranks high when it splits the branches and has an
  answer we can trust.
- **Follow-ups**: one per significant atom, ranked by MI times the binary
  entropy of the best answer's confidence, so the next questions go where the
  disagreement is large and the answer is least settled.
"""

from __future__ import annotations

import math
import re
from typing import Any

ALPHA = 0.5
_STOP = set(
    """a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each either etc few for from further had has have having he her here hers
him his how however i if in into is it its itself just let like may me might more most much my now of off on once one only
or other our ours out over own per rather same she should so some such than that the their them then there these they this
those through thus to too under until up upon us very via was we were what when where which while who whom why will with
within without would you your""".split()
)
_KEEP = {"no", "not", "nor", "never", "must", "cannot"}
_WORD = re.compile(r"[a-z][a-z0-9_'\-]*[a-z0-9]|[a-z]")


def atoms(text: str) -> set[str]:
  """Content words of 3+ letters and adjacent content-word pairs."""
  ws = [w.strip("'-") for w in _WORD.findall(text.lower())]
  uni = {w for w in ws if (len(w) >= 3 or w in _KEEP) and w not in _STOP}
  bi = {
      f"{a} {b}"
      for a, b in zip(ws, ws[1:])
      if a not in _STOP and b not in _STOP and len(a) > 1 and len(b) > 1
  }
  return uni | bi


def _entropy(p):
  return -sum(x * math.log2(x) for x in p if x > 0)


def _post(counts):
  n, k = sum(counts), len(counts)
  return [(c + ALPHA) / (n + ALPHA * k) for c in counts]


def _log_marginal(counts):
  k, n = len(counts), sum(counts)
  return (math.lgamma(ALPHA * k) - math.lgamma(ALPHA * k + n)
          + sum(math.lgamma(ALPHA + c) - math.lgamma(ALPHA) for c in counts))


def decompose(count_map: dict[str, list[int]]) -> dict[str, float]:
  """Splits total surprise into within-branch noise and I(outcome; branch)."""
  names = list(count_map)
  k = len(next(iter(count_map.values())))
  post = {m: _post(count_map[m]) for m in names}
  pbar = [sum(post[m][i] for m in names) / len(names) for i in range(k)]
  h_total = _entropy(pbar)
  h_within = sum(_entropy(post[m]) for m in names) / len(names)
  pooled = [sum(count_map[m][i] for m in names) for i in range(k)]
  log_bf = sum(_log_marginal(count_map[m]) for m in names) - _log_marginal(pooled)
  mi = max(0.0, h_total - h_within)
  return {"mi_bits": mi, "log_bf": log_bf, "share": mi / h_total if h_total else 0.0}


def confidence(votes: list[bool]) -> float:
  """Posterior mean that the answer is right, from verifier votes."""
  agree = sum(1 for v in votes if v)
  return (ALPHA + agree) / (2 * ALPHA + len(votes))


def _binary_entropy(p: float) -> float:
  return _entropy([p, 1 - p])


def analyze_question(
    node: dict[str, Any], *, mi_min_bits: float = 0.1, log_bf_min: float = 1.0, min_support: int = 2
) -> dict[str, Any]:
  """Scores one question node.

  Args:
    node: ``{"id", "question", "answers": [{"branch", "text", "votes": [bool]}]}``.
  """
  answers = node.get("answers") or []
  by_branch: dict[str, list[set[str]]] = {}
  for a in answers:
    by_branch.setdefault(a["branch"], []).append(atoms(a["text"]))
  support: dict[str, int] = {}
  for sets in by_branch.values():
    for s in sets:
      for u in s:
        support[u] = support.get(u, 0) + 1
  significant = []
  if len(by_branch) >= 2:
    for u, c in support.items():
      if c < min_support:
        continue
      cm = {b: [sum(u in s for s in sets), len(sets) - sum(u in s for s in sets)] for b, sets in by_branch.items()}
      d = decompose(cm)
      if d["mi_bits"] >= mi_min_bits and d["log_bf"] >= log_bf_min:
        rates = {b: (cm[b][0] + ALPHA) / (sum(cm[b]) + 2 * ALPHA) for b in cm}
        significant.append({"atom": u, **d, "rates": rates})
  significant.sort(key=lambda x: (-x["mi_bits"], -x["log_bf"], x["atom"]))
  scored = [
      {"branch": a["branch"], "text": a["text"], "confidence": confidence(a.get("votes") or []),
       "n_votes": len(a.get("votes") or [])}
      for a in answers
  ]
  best = max(scored, key=lambda a: (a["confidence"], a["n_votes"]), default=None)
  best_conf = best["confidence"] if best else 0.5
  total_mi = sum(a["mi_bits"] for a in significant)
  followups = [
      {
          "question": (
              f"Within '{node['question']}': which branch is right about '{a['atom']}' "
              f"(used most by {max(a['rates'], key=a['rates'].get)}, least by "
              f"{min(a['rates'], key=a['rates'].get)}), and what evidence decides it?"
          ),
          "atom": a["atom"],
          "value_of_information": a["mi_bits"] * _binary_entropy(best_conf),
      }
      for a in significant
  ]
  followups.sort(key=lambda f: -f["value_of_information"])
  return {
      "id": node["id"],
      "question": node["question"],
      "significant_atoms": significant,
      "best_answer": best,
      "question_value": total_mi * best_conf,
      "followups": followups,
  }


def refine(nodes: list[dict[str, Any]], *, top_k: int = 5, followups_per_question: int = 3, **gates) -> dict[str, Any]:
  """Keeps the top-k questions by value, each with its best follow-ups."""
  analyzed = [analyze_question(n, **gates) for n in nodes]
  ranked = sorted(analyzed, key=lambda q: (-q["question_value"], q["id"]))
  kept = [q for q in ranked if q["significant_atoms"]][:top_k]
  for q in kept:
    q["followups"] = q["followups"][:followups_per_question]
  return {
      "kept": kept,
      "dropped": [q["id"] for q in ranked if q not in kept],
      "comparisons": sum(len(q["significant_atoms"]) for q in analyzed),
  }
