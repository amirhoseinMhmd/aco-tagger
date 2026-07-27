"""A single ant walking one sentence.

Unlike the travelling-salesman problem ants are usually written for, tagging
has no "already visited" constraint: every tag stays available at every word.
An ant therefore makes exactly one weighted choice per word.
"""

from __future__ import annotations

import math
import random

from .pheromone import MIN_DIVISOR, PheromoneTable
from ..domain.solver import Solution
from ..domain.trellis import CostTrellis

# There is no predecessor before the first word; the trellis and the pheromone
# table both ignore this value at step 0.
NO_PREVIOUS_TAG = 0


class Ant:
    """Builds one candidate tagging by biased random choice."""

    def __init__(
        self,
        trellis: CostTrellis,
        pheromone: PheromoneTable,
        alpha: float,
        beta: float,
        rng: random.Random,
    ) -> None:
        self._trellis = trellis
        self._pheromone = pheromone
        self._alpha = alpha
        self._beta = beta
        self._rng = rng

    def walk(self) -> Solution:
        """Choose a tag for every word, accumulating the cost as we go."""
        path: list[int] = []
        total_cost = 0.0
        previous_tag = NO_PREVIOUS_TAG

        for step in range(self._trellis.length):
            tag = self._choose(step, previous_tag)
            total_cost += self._trellis.cost(step, previous_tag, tag)
            path.append(tag)
            previous_tag = tag

        return Solution(path=tuple(path), cost=total_cost)

    def _choose(self, step: int, previous_tag: int) -> int:
        tags = range(self._trellis.tag_count)
        weights = [self._appeal(step, previous_tag, tag) for tag in tags]

        if sum(weights) <= 0.0:
            # Every option is impossible or unappealing; stay unbiased rather
            # than always falling back to the same tag.
            return self._rng.randrange(self._trellis.tag_count)
        return self._rng.choices(tags, weights=weights)[0]

    def _appeal(self, step: int, previous_tag: int, tag: int) -> float:
        """``pheromone^alpha * visibility^beta`` - the standard ACO weighting.

        ``alpha`` weighs what the colony learned, ``beta`` weighs the local
        cost. Impossible edges score zero and are never taken.
        """
        cost = self._trellis.cost(step, previous_tag, tag)
        if math.isinf(cost):
            return 0.0
        visibility = 1.0 / max(cost, MIN_DIVISOR)
        level = self._pheromone.level(step, previous_tag, tag)
        return level**self._alpha * visibility**self._beta
