"""Pheromone storage and the strategies that reinforce it.

The table is indexed exactly like :class:`~pos_aco.trellis.CostTrellis`, by
``(step, previous_tag, tag)``, so a trail always refers to the same edge as the
cost it was earned on.

Adding a new reinforcement scheme means adding a :class:`DepositStrategy`; the
colony itself does not change.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from ..domain.solver import Solution
from ..domain.trellis import CostTrellis

Row = list[float]

# Guards the reciprocals below: a probability of 1 costs 0, which would
# otherwise make a deposit or a heuristic value infinite.
MIN_DIVISOR = 1e-12


class PheromoneTable:
    """Mutable trail levels for one trellis."""

    def __init__(self, trellis: CostTrellis, initial_level: float = 1.0) -> None:
        if initial_level <= 0.0:
            raise ValueError(f"initial pheromone must be positive, got {initial_level}")
        tag_count = trellis.tag_count
        self._start: Row = [initial_level] * tag_count
        self._steps: list[list[Row]] = [
            [[initial_level] * tag_count for _ in range(tag_count)]
            for _ in range(trellis.length - 1)
        ]

    def level(self, step: int, previous_tag: int, tag: int) -> float:
        return self._row(step, previous_tag)[tag]

    def deposit(self, step: int, previous_tag: int, tag: int, amount: float) -> None:
        self._row(step, previous_tag)[tag] += amount

    def evaporate(self, rate: float) -> None:
        """Scale every trail down by ``rate`` (0 keeps everything, 1 wipes it)."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"evaporation rate must be in [0, 1], got {rate}")
        retained = 1.0 - rate
        for row in self._rows():
            for tag in range(len(row)):
                row[tag] *= retained

    def _row(self, step: int, previous_tag: int) -> Row:
        """The trails leaving ``previous_tag`` at ``step``.

        Step 0 has no predecessor, so all ants share a single row - the same
        convention the trellis uses.
        """
        if step == 0:
            return self._start
        return self._steps[step - 1][previous_tag]

    def _rows(self) -> Iterator[Row]:
        yield self._start
        for matrix in self._steps:
            yield from matrix


class DepositStrategy(Protocol):
    """Decides how much pheromone one ant leaves on the edges it used."""

    def deposit(
        self, table: PheromoneTable, trellis: CostTrellis, solution: Solution
    ) -> None: ...


def edges(solution: Solution) -> Iterator[tuple[int, int, int]]:
    """The ``(step, previous_tag, tag)`` triples a solution travels through."""
    previous_tag = 0
    for step, tag in enumerate(solution.path):
        yield step, previous_tag, tag
        previous_tag = tag


class AntCycle:
    """Reinforce by overall path quality: ``Q / total_cost`` on every edge.

    The classic Ant System rule, and the default: an ant that found a good
    sentence tagging strengthens all of its choices equally.
    """

    def __init__(self, q: float = 1.0) -> None:
        self._q = q

    def deposit(
        self, table: PheromoneTable, trellis: CostTrellis, solution: Solution
    ) -> None:
        amount = self._q / max(solution.cost, MIN_DIVISOR)
        for step, previous_tag, tag in edges(solution):
            table.deposit(step, previous_tag, tag, amount)


class AntDensity:
    """Reinforce every used edge by a constant ``Q``, ignoring quality."""

    def __init__(self, q: float = 1.0) -> None:
        self._q = q

    def deposit(
        self, table: PheromoneTable, trellis: CostTrellis, solution: Solution
    ) -> None:
        for step, previous_tag, tag in edges(solution):
            table.deposit(step, previous_tag, tag, self._q)


class AntQuantity:
    """Reinforce each edge by ``Q / edge_cost``, rewarding cheap edges locally."""

    def __init__(self, q: float = 1.0) -> None:
        self._q = q

    def deposit(
        self, table: PheromoneTable, trellis: CostTrellis, solution: Solution
    ) -> None:
        for step, previous_tag, tag in edges(solution):
            cost = trellis.cost(step, previous_tag, tag)
            table.deposit(step, previous_tag, tag, self._q / max(cost, MIN_DIVISOR))
