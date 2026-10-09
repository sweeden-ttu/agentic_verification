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

"""Counterfactual regret minimization (CFR+) for two-player zero-sum games.

The loop uses CFR to choose what to improve when the payoff of each choice
depends on something it does not control: which evaluation slice the score is
measured on, or which follow-up answer turns out to hold. Modelling that as an
adversary and solving for the equilibrium gives the policy with the best
worst-case improvement, rather than the one that looks best on average.

``solve`` runs CFR+ (regret matching with non-negative cumulative regrets and
linearly weighted averaging) on any game exposing the ``Game`` protocol below.
The average strategy converges to a Nash equilibrium; ``exploitability``
measures how far it is from one. ``solve_matrix`` is the one-shot special case.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol, Sequence

CHANCE = -1


class Game(Protocol):
  """A two-player zero-sum extensive-form game with chance."""

  def root(self) -> Any: ...
  def is_terminal(self, h: Any) -> bool: ...
  def utility(self, h: Any) -> float: ...  # payoff to player 0
  def player(self, h: Any) -> int: ...  # 0, 1 or CHANCE
  def chance_outcomes(self, h: Any) -> Sequence[tuple[Any, float]]: ...
  def infoset(self, h: Any) -> Hashable: ...
  def actions(self, h: Any) -> Sequence[Any]: ...
  def next(self, h: Any, a: Any) -> Any: ...


def _match(regrets: list[float]) -> list[float]:
  pos = [max(r, 0.0) for r in regrets]
  total = sum(pos)
  n = len(regrets)
  return [p / total for p in pos] if total > 0 else [1.0 / n] * n


class CFRSolver:
  """CFR+ with alternating updates."""

  def __init__(self, game: Game):
    self.game = game
    self.regret: dict[Hashable, list[float]] = {}
    self.strategy_sum: dict[Hashable, list[float]] = {}
    self.actions: dict[Hashable, list[Any]] = {}
    self.iterations = 0

  def _node(self, h: Any) -> Hashable:
    key = self.game.infoset(h)
    if key not in self.regret:
      acts = list(self.game.actions(h))
      self.actions[key] = acts
      self.regret[key] = [0.0] * len(acts)
      self.strategy_sum[key] = [0.0] * len(acts)
    return key

  def _cfr(self, h: Any, traverser: int, reach: tuple[float, float], weight: float) -> float:
    g = self.game
    if g.is_terminal(h):
      u = g.utility(h)
      return u if traverser == 0 else -u
    p = g.player(h)
    if p == CHANCE:
      return sum(pr * self._cfr(g.next(h, o), traverser, reach, weight) for o, pr in g.chance_outcomes(h))
    key = self._node(h)
    sigma = _match(self.regret[key])
    acts = self.actions[key]
    values = []
    for a, s in zip(acts, sigma):
      r = (reach[0] * s, reach[1]) if p == 0 else (reach[0], reach[1] * s)
      values.append(self._cfr(g.next(h, a), traverser, r, weight))
    node_value = sum(s * v for s, v in zip(sigma, values))
    if p == traverser:
      opp = reach[1 - p]
      self.regret[key] = [max(0.0, rg + opp * (v - node_value)) for rg, v in zip(self.regret[key], values)]
    else:
      own = reach[p]
      self.strategy_sum[key] = [ss + weight * own * s for ss, s in zip(self.strategy_sum[key], sigma)]
    return node_value

  def iterate(self, n: int) -> None:
    for _ in range(n):
      self.iterations += 1
      for traverser in (0, 1):
        self._cfr(self.game.root(), traverser, (1.0, 1.0), float(self.iterations))

  def average_strategy(self) -> dict[Hashable, dict[Any, float]]:
    out = {}
    for key, ss in self.strategy_sum.items():
      total = sum(ss)
      probs = [s / total for s in ss] if total > 0 else [1.0 / len(ss)] * len(ss)
      out[key] = dict(zip(self.actions[key], probs))
    return out


def expected_value(game: Game, strategy: dict[Hashable, dict[Any, float]], h: Any = None) -> float:
  """Value to player 0 when both players follow ``strategy``."""
  h = game.root() if h is None else h
  if game.is_terminal(h):
    return game.utility(h)
  if game.player(h) == CHANCE:
    return sum(pr * expected_value(game, strategy, game.next(h, o)) for o, pr in game.chance_outcomes(h))
  acts = list(game.actions(h))
  sigma = strategy.get(game.infoset(h)) or {a: 1.0 / len(acts) for a in acts}
  return sum(sigma.get(a, 0.0) * expected_value(game, strategy, game.next(h, a)) for a in acts)


def _best_response_value(game: Game, strategy, responder: int) -> float:
  """Value to ``responder`` of a best response to ``strategy``."""
  # Collect, per responder infoset, the opponent/chance-weighted value of each action.
  from collections import defaultdict

  def walk(h, reach, plan):
    if game.is_terminal(h):
      u = game.utility(h)
      return reach * (u if responder == 0 else -u)
    p = game.player(h)
    if p == CHANCE:
      return sum(walk(game.next(h, o), reach * pr, plan) for o, pr in game.chance_outcomes(h))
    acts = list(game.actions(h))
    if p == responder:
      a = plan.get(game.infoset(h), acts[0])
      return walk(game.next(h, a), reach, plan)
    sigma = strategy.get(game.infoset(h)) or {a: 1.0 / len(acts) for a in acts}
    return sum(walk(game.next(h, a), reach * sigma.get(a, 0.0), plan) for a in acts)

  # Iterate best-response improvement over infosets until stable (exact for
  # perfect-recall games of the small sizes the loop builds).
  plan: dict[Hashable, Any] = {}
  infosets = defaultdict(list)

  def collect(h):
    if game.is_terminal(h):
      return
    if game.player(h) == CHANCE:
      for o, _ in game.chance_outcomes(h):
        collect(game.next(h, o))
      return
    if game.player(h) == responder:
      infosets[game.infoset(h)] = list(game.actions(h))
    for a in game.actions(h):
      collect(game.next(h, a))

  collect(game.root())
  best = walk(game.root(), 1.0, plan)
  changed = True
  while changed:
    changed = False
    for key, acts in infosets.items():
      for a in acts:
        trial = {**plan, key: a}
        v = walk(game.root(), 1.0, trial)
        if v > best + 1e-12:
          plan, best, changed = trial, v, True
  return best


def exploitability(game: Game, strategy) -> float:
  """Mean gain of a best-responding opponent; 0 at a Nash equilibrium."""
  return (_best_response_value(game, strategy, 0) + _best_response_value(game, strategy, 1)) / 2


class MatrixGame:
  """One simultaneous move: row player 0 maximises ``payoff[i][j]``."""

  def __init__(self, payoff: Sequence[Sequence[float]], rows=None, cols=None):
    self.payoff = [list(r) for r in payoff]
    self.rows = list(rows or range(len(payoff)))
    self.cols = list(cols or range(len(payoff[0])))

  def root(self):
    return ()

  def is_terminal(self, h):
    return len(h) == 2

  def utility(self, h):
    return self.payoff[self.rows.index(h[0])][self.cols.index(h[1])]

  def player(self, h):
    return len(h)

  def chance_outcomes(self, h):
    return []

  def infoset(self, h):
    return ("row",) if len(h) == 0 else ("col",)  # columns do not see the row

  def actions(self, h):
    return self.rows if len(h) == 0 else self.cols

  def next(self, h, a):
    return h + (a,)


def solve(game: Game, iterations: int = 2000) -> tuple[dict, float]:
  """Returns the average strategy and its value to player 0."""
  solver = CFRSolver(game)
  solver.iterate(iterations)
  strategy = solver.average_strategy()
  return strategy, expected_value(game, strategy)


def solve_matrix(payoff, rows=None, cols=None, iterations: int = 2000) -> dict[str, Any]:
  """Robust policy for a payoff matrix of row choices against column outcomes."""
  game = MatrixGame(payoff, rows, cols)
  strategy, value = solve(game, iterations)
  return {
      "row_policy": strategy[("row",)],
      "col_policy": strategy[("col",)],
      "value": value,
      "exploitability": exploitability(game, strategy),
  }
