"""The abstraction every search strategy implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .trellis import CostTrellis


@dataclass(frozen=True)
class Solution:
    """A complete tag assignment and what it costs."""

    path: tuple[int, ...]
    cost: float


class PathSolver(Protocol):
    """Finds a low-cost path through a trellis.

    Implementations differ in whether they guarantee the optimum
    (:class:`~pos_aco.viterbi.ViterbiSolver`) or approximate it
    (:class:`~pos_aco.colony.AntColonySolver`).
    """

    def solve(self, trellis: CostTrellis) -> Solution: ...
