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

"""Tests for the CFR+ solver."""

from __future__ import annotations

import itertools

from knowledge_loop import cfr


class Kuhn:
  """Kuhn poker: 3 cards, ante 1, one bet of 1. Value to player 0 is -1/18."""

  def root(self):
    return ()

  def is_terminal(self, h):
    acts = "".join(h[1:]) if h else ""
    return bool(h) and acts in ("pp", "bp", "bb", "pbp", "pbb")

  def utility(self, h):
    cards, acts = h[0], "".join(h[1:])
    win = 1 if cards[0] > cards[1] else -1
    if acts == "bp":
      return 1
    if acts == "pbp":
      return -1
    return win * (2 if acts in ("bb", "pbb") else 1)

  def player(self, h):
    return cfr.CHANCE if not h else (len(h) - 1) % 2

  def chance_outcomes(self, h):
    deals = list(itertools.permutations([0, 1, 2], 2))
    return [(d, 1 / len(deals)) for d in deals]

  def infoset(self, h):
    p = self.player(h)
    return (h[0][p], "".join(h[1:]))

  def actions(self, h):
    return ["p", "b"]

  def next(self, h, a):
    return h + (a,)


def test_kuhn_poker_reaches_the_known_game_value():
  """CFR+ finds Kuhn poker's equilibrium value of -1/18 for player 0."""
  strategy, value = cfr.solve(Kuhn(), iterations=3000)
  assert abs(value - (-1 / 18)) < 2e-3


def test_kuhn_poker_strategy_is_nearly_unexploitable():
  """A best-responding opponent gains almost nothing against the average strategy."""
  strategy, _ = cfr.solve(Kuhn(), iterations=3000)
  assert cfr.exploitability(Kuhn(), strategy) < 5e-3


def test_rock_paper_scissors_equilibrium_is_uniform():
  """The unique equilibrium of rock-paper-scissors mixes evenly."""
  rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
  result = cfr.solve_matrix(rps, iterations=3000)
  assert all(abs(p - 1 / 3) < 0.02 for p in result["row_policy"].values())
  assert abs(result["value"]) < 0.01


def test_matrix_policy_prefers_the_robust_choice():
  """A choice that is never bad beats one that is great on one slice only."""
  payoff = [[0.30, -0.20], [0.05, 0.04]]  # rows: risky, steady; cols: slices
  result = cfr.solve_matrix(payoff, rows=["risky", "steady"], cols=["s1", "s2"])
  assert result["row_policy"]["steady"] > 0.95
  assert abs(result["value"] - 0.04) < 0.01
