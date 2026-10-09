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

"""Tests for the trace-language spec and the prophecy variable."""

from __future__ import annotations

import copy
import datetime
import json

from lora_improvement_loop import spec
import pytest


def _run(mon: spec.SpecMonitor) -> None:
  mon.train_epoch()
  mon.evaluate(2)
  mon.package()
  mon.tick()
  mon.train_epoch()
  mon.evaluate(3)


def test_monitor_trace_is_in_the_spec_language():
  """A trace recorded by the monitor replays with no violations."""
  mon = spec.SpecMonitor(prophecy="win")
  _run(mon)
  assert spec.check_trace(mon.s["trace"]) == []


def test_train_epoch_is_refused_when_the_budget_is_spent():
  """TrainEpoch needs budget, a train phase and an open deadline."""
  mon = spec.SpecMonitor(prophecy="win")
  for _ in range(spec.WEEKLY_BUDGET):
    mon.train_epoch()
    mon.evaluate(1)
  with pytest.raises(spec.SpecViolation):
    mon.train_epoch()


def test_budget_refills_every_seventh_day():
  """Tick sets the budget back to the weekly value on day 7."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  mon.evaluate(1)
  for _ in range(7):
    mon.tick()
  assert (mon.s["day"], mon.s["budget"]) == (7, spec.WEEKLY_BUDGET)


def test_evaluate_rejects_a_score_outside_the_range():
  """The score must be an integer from 0 to MAX_SCORE."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  with pytest.raises(spec.SpecViolation):
    mon.evaluate(spec.MAX_SCORE + 1)


def test_submit_needs_human_confirmation():
  """Submit is refused unless a human confirmed it."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  mon.evaluate(2)
  mon.package()
  with pytest.raises(spec.SpecViolation):
    mon.submit(confirmed_by_user=False)
  mon.submit(confirmed_by_user=True)
  assert mon.s["sub"] == 2


def test_check_trace_flags_training_past_the_budget():
  """Forging a fourth epoch in one week is reported with its step number."""
  mon = spec.SpecMonitor(prophecy="win")
  for _ in range(spec.WEEKLY_BUDGET):
    mon.train_epoch()
    mon.evaluate(1)
  forged = copy.deepcopy(mon.s["trace"])
  forged.append({**forged[-1], "action": "TrainEpoch", "phase": "eval", "budget": -1})
  problems = spec.check_trace(forged)
  assert problems and problems[0].startswith(f"step {len(forged) - 1}: TrainEpoch")


def test_check_trace_flags_a_state_the_step_does_not_produce():
  """A recorded successor that differs from the spec's is reported."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  forged = copy.deepcopy(mon.s["trace"])
  forged[-1]["budget"] = 3
  assert any("produced" in p for p in spec.check_trace(forged))


def test_check_trace_flags_a_changed_prophecy():
  """The prophecy is constant, so flipping it mid-trace is a violation."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  forged = copy.deepcopy(mon.s["trace"])
  forged[-1]["p"] = "lose"
  assert spec.check_trace(forged)


def test_prophecy_is_a_ghost_value():
  """Runs differing only in p enable the same actions and reach the same states."""
  runs = []
  for p in ("win", "lose"):
    mon = spec.SpecMonitor(prophecy=p)
    seen = []
    for act in (mon.train_epoch, lambda m=mon: m.evaluate(2), mon.package, mon.tick):
      seen.append({a: mon.enabled(a) for a in spec.ACTIONS if a != "Init" and not a.startswith("Judge")})
      act()
    runs.append((seen, [{k: v for k, v in t.items() if k not in ("p", "at")} for t in mon.s["trace"]]))
  assert runs[0] == runs[1]


def test_judge_win_needs_a_submitted_score_at_the_win_bar():
  """JudgeWin is enabled only after the deadline and sub >= WIN_BAR."""
  mon = spec.SpecMonitor(prophecy="win", today=spec.START + datetime.timedelta(days=60))
  assert mon.s["day"] == spec.DEADLINE_DAY + 1
  assert mon.enabled("JudgeLose") and not mon.enabled("JudgeWin")
  with pytest.raises(spec.SpecViolation):
    mon.judge(won=True)


def test_judge_reports_whether_the_prophecy_came_true():
  """The verdict compares the outcome with p."""
  mon = spec.SpecMonitor(prophecy="lose", today=spec.START + datetime.timedelta(days=60))
  verdict = mon.judge(won=False)
  assert verdict == {"prophecy": "lose", "outcome": "lose", "prophecy_fulfilled": True}


def test_catch_up_stops_at_the_deadline():
  """The monitor never ticks beyond DEADLINE_DAY + 1."""
  mon = spec.SpecMonitor(prophecy="win", today=spec.START + datetime.timedelta(days=500))
  assert mon.s["day"] == spec.DEADLINE_DAY + 1


def test_state_file_round_trips_in_the_notebook_layout(tmp_path):
  """A saved state file holds the spec variables plus a trace list."""
  path = tmp_path / "spec_state.json"
  _run(spec.SpecMonitor(str(path), prophecy="win"))
  saved = json.loads(path.read_text())
  assert set(spec.STATE_VARS) | {"trace"} == set(saved)
  assert spec.SpecMonitor(str(path)).view() == spec.view(saved)


def test_trace_extends_ignores_timestamps():
  """A prefix matches on action and variables, not on the 'at' field."""
  mon = spec.SpecMonitor(prophecy="win")
  mon.train_epoch()
  old = copy.deepcopy(mon.s["trace"])
  mon.evaluate(1)
  for entry in old:
    entry["at"] = "1999-01-01T00:00:00"
  assert spec.trace_extends(old, mon.s["trace"])
  assert not spec.trace_extends(mon.s["trace"], old)
