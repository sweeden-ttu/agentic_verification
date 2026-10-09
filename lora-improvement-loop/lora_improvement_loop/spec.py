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

"""Trace-language specification of the LoRA training loop, with a prophecy.

The module is a Python port of the transition relation in the baseline
notebook (``gemma4-lora-prophecy-wet-31b``), which in turn mirrors the TLA+
spec ``Gemma4Agent.tla``. A *behavior* is a sequence of states. A *trace* is
the sequence of ``{"action": ..., **state}`` records the notebook appends to
``spec_state.json``. The language of the spec is the set of traces whose every
step is enabled and whose successor state matches ``step``.

``step`` and ``enabled`` are pure functions, so the same transition relation
drives both the live ``SpecMonitor`` and ``check_trace``, which replays a
recorded trace and reports every step the spec does not allow.

The prophecy variable ``p`` follows Lamport and Merz, *Prophecy Made Simple*,
section 4.2. It predicts which ``Judge`` action happens next. It is a ghost
value: it is stored and reported, and no guard or transition reads it.
"""

from __future__ import annotations

import datetime
import json
import os
import random
from typing import Any

START = datetime.date(2026, 10, 8)
DEADLINE_DAY = 55
WEEKLY_BUDGET = 3
MAX_SCORE = 3
WIN_BAR = 3
SCORE_THRESHOLDS = (0.25, 0.50, 0.75)

STATE_VARS = (
    "day",
    "phase",
    "budget",
    "score",
    "best",
    "sub",
    "outcome",
    "p",
)
ACTIONS = (
    "Init",
    "TrainEpoch",
    "Evaluate",
    "Package",
    "Submit",
    "Tick",
    "JudgeWin",
    "JudgeLose",
)


class SpecViolation(Exception):
  """Raised when an action is not enabled in the current state."""


def initial_state(prophecy: str) -> dict[str, Any]:
  """Returns the ``Init`` state of the spec."""
  if prophecy not in ("win", "lose"):
    raise ValueError(f"prophecy must be 'win' or 'lose', got {prophecy!r}")
  return dict(
      day=0,
      phase="train",
      budget=WEEKLY_BUDGET,
      score=0,
      best=0,
      sub=0,
      outcome="pending",
      p=prophecy,
  )


def score_from_accuracy(accuracy: float) -> int:
  """Maps teacher-forced next-tool accuracy to the spec's ``score`` in 0..3."""
  return sum(accuracy >= t for t in SCORE_THRESHOLDS)


def _is_open(state: dict[str, Any]) -> bool:
  return state["day"] <= DEADLINE_DAY


def enabled(state: dict[str, Any], action: str) -> bool:
  """Returns whether ``action`` is enabled in ``state``.

  No guard reads ``state["p"]``. That is what keeps ``p`` a ghost value.
  """
  is_open = _is_open(state)
  phase = state["phase"]
  judging = (not is_open) and state["outcome"] == "pending"
  if action == "TrainEpoch":
    return is_open and phase == "train" and state["budget"] > 0
  if action == "Evaluate":
    return phase == "eval"
  if action == "Package":
    return is_open and phase == "train" and state["best"] > state["sub"]
  if action == "Submit":
    return is_open and phase == "packaged"
  if action == "Tick":
    return is_open
  if action == "Judge":
    return judging
  if action == "JudgeWin":
    return judging and state["sub"] >= WIN_BAR
  if action == "JudgeLose":
    return judging
  raise KeyError(f"unknown action {action!r}")


def step(
    state: dict[str, Any],
    action: str,
    *,
    score: int | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
  """Returns the successor state, or raises ``SpecViolation``.

  Args:
    state: The current state; it is not modified.
    action: One of ``TrainEpoch``, ``Evaluate``, ``Package``, ``Submit``,
      ``Tick``, ``JudgeWin`` or ``JudgeLose``.
    score: The measured score, required for ``Evaluate``.
    confirmed: Whether a human confirmed the submission, required for
      ``Submit``.
  """
  if not enabled(state, action):
    raise SpecViolation(f"{action} not enabled in state {view(state)}")
  new = {k: state[k] for k in STATE_VARS}
  if action == "TrainEpoch":
    new["budget"] -= 1
    new["phase"] = "eval"
  elif action == "Evaluate":
    if not isinstance(score, int) or not 0 <= score <= MAX_SCORE:
      raise SpecViolation(f"Evaluate.score must be in 0..{MAX_SCORE}: {score!r}")
    new["score"] = score
    new["best"] = max(new["best"], score)
    new["phase"] = "train"
  elif action == "Package":
    new["phase"] = "packaged"
  elif action == "Submit":
    if not confirmed:
      raise SpecViolation("Submit requires confirmation by a human")
    new["sub"] = new["best"]
    new["phase"] = "train"
  elif action == "Tick":
    new["day"] += 1
    if new["day"] % 7 == 0:
      new["budget"] = WEEKLY_BUDGET
    if new["phase"] == "packaged":
      new["phase"] = "train"
  elif action == "JudgeWin":
    new["outcome"] = "win"
  elif action == "JudgeLose":
    new["outcome"] = "lose"
  return new


def view(state: dict[str, Any]) -> dict[str, Any]:
  """Returns the spec variables of a state record, dropping ``trace``."""
  return {k: state[k] for k in STATE_VARS}


def check_trace(trace: list[dict[str, Any]]) -> list[str]:
  """Replays a recorded trace and returns one message per violation.

  An empty list means the trace is in the language of the spec. The check
  covers the initial state, enabledness of every action, the successor state
  of every step, and that the prophecy never changes.

  Args:
    trace: The ``trace`` list of a ``spec_state.json``.
  """
  problems: list[str] = []
  if not trace:
    return ["trace is empty"]
  first = trace[0]
  if first.get("action") != "Init":
    problems.append(f"step 0: first action is {first.get('action')!r}, not Init")
  try:
    expected = initial_state(first.get("p"))
  except ValueError as e:
    return problems + [f"step 0: {e}"]
  if view(first) != expected:
    problems.append(f"step 0: Init state {view(first)} != {expected}")
  prev = view(first)
  for i, entry in enumerate(trace[1:], start=1):
    action = entry.get("action")
    if action not in ACTIONS or action == "Init":
      problems.append(f"step {i}: unknown or repeated action {action!r}")
      continue
    try:
      expected = step(
          prev,
          action,
          score=entry.get("score") if action == "Evaluate" else None,
          confirmed=True,
      )
    except SpecViolation as e:
      problems.append(f"step {i}: {e}")
      prev = view(entry)
      continue
    if view(entry) != expected:
      problems.append(
          f"step {i}: {action} produced {view(entry)}, spec allows {expected}"
      )
    prev = view(entry)
  return problems


def trace_extends(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> bool:
  """Returns whether ``old`` is a prefix of ``new`` ignoring timestamps."""
  if len(old) > len(new):
    return False
  return all(
      a.get("action") == b.get("action") and view(a) == view(b)
      for a, b in zip(old, new)
  )


def prophecy_report(trace: list[dict[str, Any]]) -> dict[str, Any]:
  """Reports whether the prophecy matched the judged outcome, if any."""
  last = trace[-1]
  outcome = last["outcome"]
  return {
      "prophecy": last["p"],
      "outcome": outcome,
      "prophecy_fulfilled": None if outcome == "pending" else outcome == last["p"],
  }


class SpecMonitor:
  """Runtime monitor holding one behavior of the spec.

  A monitor with a ``path`` loads and saves the same JSON layout the baseline
  notebook writes to ``spec_state.json``, so either side can continue the
  other's behavior. A monitor without a ``path`` lives in memory only.
  """

  def __init__(
      self,
      path: str | None = None,
      *,
      state: dict[str, Any] | None = None,
      prophecy: str | None = None,
      rng: random.Random | None = None,
      today: datetime.date | None = None,
  ):
    self.path = path
    if state is not None:
      self.s = state
    elif path and os.path.exists(path):
      with open(path, encoding="utf-8") as f:
        self.s = json.load(f)
    else:
      p = prophecy or (rng or random).choice(["win", "lose"])
      self.s = {**initial_state(p), "trace": []}
      self._log("Init")
    if today is not None:
      self.catch_up(today)

  def catch_up(self, today: datetime.date) -> None:
    """Applies ``Tick`` until ``day`` matches the calendar, capped at the deadline."""
    horizon = datetime.timedelta(days=DEADLINE_DAY + 1)
    target = min(today - START, horizon).days
    while self.s["day"] < max(target, 0):
      self.tick()

  def view(self) -> dict[str, Any]:
    return view(self.s)

  def enabled(self, action: str) -> bool:
    return enabled(self.s, action)

  def _apply(self, action: str, **kwargs: Any) -> None:
    self.s.update(step(self.s, action, **kwargs))
    self._log(action)

  def _log(self, action: str) -> None:
    self.s["trace"].append({
        "action": action,
        **self.view(),
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    if self.path:
      with open(self.path, "w", encoding="utf-8") as f:
        json.dump(self.s, f, indent=1)

  def train_epoch(self) -> None:
    self._apply("TrainEpoch")

  def evaluate(self, new_score: int) -> None:
    self._apply("Evaluate", score=new_score)

  def package(self) -> None:
    self._apply("Package")

  def submit(self, confirmed_by_user: bool) -> None:
    self._apply("Submit", confirmed=confirmed_by_user)

  def tick(self) -> None:
    self._apply("Tick")

  def judge(self, won: bool) -> dict[str, Any]:
    self._apply("JudgeWin" if won else "JudgeLose")
    return prophecy_report(self.s["trace"])
