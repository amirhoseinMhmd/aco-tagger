"""Ant colony optimisation over a tag trellis."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ant import Ant
from .pheromone import AntCycle, DepositStrategy, PheromoneTable
from ..domain.solver import Solution
from ..domain.trellis import CostTrellis


@dataclass(frozen=True)
class ColonyParameters:
    """Tuning knobs for :class:`AntColonySolver`.

    ``alpha`` weighs pheromone against ``beta``'s weighting of raw cost;
    ``evaporation`` is the fraction of every trail lost per generation.
    """

    ant_count: int = 20
    generations: int = 50
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.1
    initial_pheromone: float = 1.0

    def __post_init__(self) -> None:
        if self.ant_count < 1:
            raise ValueError(f"ant_count must be at least 1, got {self.ant_count}")
        if self.generations < 1:
            raise ValueError(f"generations must be at least 1, got {self.generations}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0.0 <= self.evaporation <= 1.0:
            raise ValueError(
                f"evaporation must be in [0, 1], got {self.evaporation}"
            )
        if self.initial_pheromone <= 0.0:
            raise ValueError("initial_pheromone must be positive")


# Table 4 of the ACO-tagger paper, reported there as the best combination
# found by grid search. Note two gaps the paper leaves open:
#   * it states the pheromone array starts at zero, which makes its equation
#     (2) a 0/0 in the first generation, so a positive initial level is needed;
#   * its equation (3) sums an unspecified deposit term, so the Q = 10 it lists
#     has to be paired with a choice of DepositStrategy.
PAPER_PARAMETERS = ColonyParameters(
    ant_count=20,
    generations=3,
    alpha=0.9,
    beta=0.9,
    evaporation=0.95,
)

PAPER_DEPOSIT_FACTOR = 10.0


class AntColonySolver:
    """Approximates the cheapest path by repeated, pheromone-guided sampling.

    The deposit strategy and the random source are injected, so reinforcement
    schemes can be swapped and runs can be made reproducible in tests.
    """

    def __init__(
        self,
        parameters: ColonyParameters | None = None,
        deposit: DepositStrategy | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._parameters = parameters or ColonyParameters()
        self._deposit = deposit or AntCycle()
        self._rng = rng or random.Random()

    def solve(self, trellis: CostTrellis) -> Solution:
        parameters = self._parameters
        pheromone = PheromoneTable(trellis, parameters.initial_pheromone)
        best: Solution | None = None

        for _ in range(parameters.generations):
            solutions = [self._release_ant(trellis, pheromone) for _ in range(parameters.ant_count)]

            pheromone.evaporate(parameters.evaporation)
            for solution in solutions:
                self._deposit.deposit(pheromone, trellis, solution)
                if best is None or solution.cost < best.cost:
                    best = solution

        assert best is not None  # guaranteed: generations and ant_count are >= 1
        return best

    def _release_ant(
        self, trellis: CostTrellis, pheromone: PheromoneTable
    ) -> Solution:
        ant = Ant(
            trellis=trellis,
            pheromone=pheromone,
            alpha=self._parameters.alpha,
            beta=self._parameters.beta,
            rng=self._rng,
        )
        return ant.walk()
