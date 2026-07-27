"""How a cheap path through the trellis is found.

Depends on :mod:`pos_aco.domain` only. Every solver here satisfies
:class:`~pos_aco.domain.solver.PathSolver`, so they are interchangeable - which
is what lets the tests check the colony against the exact answer.
"""

from .ant import Ant
from .colony import (
    PAPER_DEPOSIT_FACTOR,
    PAPER_PARAMETERS,
    AntColonySolver,
    ColonyParameters,
)
from .pheromone import (
    AntCycle,
    AntDensity,
    AntQuantity,
    DepositStrategy,
    PheromoneTable,
    edges,
)
from .viterbi import ViterbiSolver

__all__ = [
    "PAPER_DEPOSIT_FACTOR",
    "PAPER_PARAMETERS",
    "Ant",
    "AntColonySolver",
    "AntCycle",
    "AntDensity",
    "AntQuantity",
    "ColonyParameters",
    "DepositStrategy",
    "PheromoneTable",
    "ViterbiSolver",
    "edges",
]
