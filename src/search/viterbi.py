"""Exact dynamic-programming solver.

Kept alongside the ant colony as a reference: it is what the colony is trying
to approximate, and the test suite uses it as the ground truth.
"""

from __future__ import annotations

from ..domain.solver import Solution
from ..domain.trellis import CostTrellis


class ViterbiSolver:
    """Finds the guaranteed cheapest path in ``O(words * tags^2)`` time."""

    def solve(self, trellis: CostTrellis) -> Solution:
        tag_count = trellis.tag_count
        best = [trellis.cost(0, 0, tag) for tag in range(tag_count)]
        backpointers: list[list[int]] = []

        for step in range(1, trellis.length):
            previous = best
            best = []
            choices = []
            for tag in range(tag_count):
                source = min(
                    range(tag_count),
                    key=lambda prev: previous[prev] + trellis.cost(step, prev, tag),
                )
                best.append(previous[source] + trellis.cost(step, source, tag))
                choices.append(source)
            backpointers.append(choices)

        final = min(range(tag_count), key=lambda tag: best[tag])
        path = [final]
        for choices in reversed(backpointers):
            path.append(choices[path[-1]])
        path.reverse()

        return Solution(path=tuple(path), cost=best[final])
