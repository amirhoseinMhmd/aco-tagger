"""The search space: a cost per (position, previous tag, tag) triple.

An emission and a transition probability are combined into a single additive
edge cost, so the best tagging is the cheapest path and every solver can work
with plain sums.

Two cost functions ship, and they are *not* equivalent - see
:func:`paper_distance` for the difference and why it matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .model import LanguageModel

Row = tuple[float, ...]
Matrix = tuple[Row, ...]


class CostFunction(Protocol):
    """Turns one ``(emission, transition)`` pair into an edge cost."""

    def __call__(self, emission: float, transition: float) -> float: ...


def to_cost(probability: float) -> float:
    """``-log10(probability)``, with impossible events costing ``inf``."""
    if probability <= 0.0:
        return math.inf
    return -math.log10(probability)


def negative_log_likelihood(emission: float, transition: float) -> float:
    """``-log10(emission * transition)`` - the standard HMM path cost.

    Summing this over a path is the same as maximising the joint probability
    of the tagging, which is exactly what the Viterbi algorithm optimises.
    """
    return to_cost(emission * transition)


def paper_distance(emission: float, transition: float) -> float:
    """``emission ** log10(transition)`` - equation (1) of the ACO-tagger paper.

    This is what the paper specifies, and it is *not* the HMM log-likelihood.
    Two consequences are worth knowing before using it:

    * Because ``x ** 0 == 1``, an edge whose emission probability is 1 costs
      exactly 1 no matter how improbable its transition is. Unambiguous words
      therefore ignore the transition model entirely - the very mechanism an
      HMM uses to disambiguate.
    * The cost is ``10 ** (log10(e) * log10(t))``, a product of surprisals in
      the exponent, so its range explodes: real sentences produce finite edge
      costs spanning 1 to over 1e11, and a single edge can dominate the sum.

    Kept as the default for fidelity to the published method. See
    :func:`negative_log_likelihood` for the conventional alternative.
    """
    if emission <= 0.0 or transition <= 0.0:
        return math.inf
    return emission ** math.log10(transition)


@dataclass(frozen=True)
class CostTrellis:
    """Costs for tagging one sentence.

    The first word has no predecessor, so its costs are stored separately
    instead of being padded with unreachable rows. :meth:`cost` hides that
    split, giving solvers a single uniform lookup.
    """

    tags: tuple[str, ...]
    start_costs: Row
    step_costs: tuple[Matrix, ...]

    @property
    def length(self) -> int:
        """Number of words in the sentence."""
        return 1 + len(self.step_costs)

    @property
    def tag_count(self) -> int:
        return len(self.tags)

    def cost(self, step: int, previous_tag: int, tag: int) -> float:
        """Cost of assigning ``tag`` at ``step``, coming from ``previous_tag``.

        ``previous_tag`` is ignored at step 0, where there is no predecessor.
        """
        if step == 0:
            return self.start_costs[tag]
        return self.step_costs[step - 1][previous_tag][tag]

    def total_cost(self, path: Sequence[int]) -> float:
        """Cost of a complete tag assignment."""
        if len(path) != self.length:
            raise ValueError(
                f"path has {len(path)} tags, sentence has {self.length} words"
            )
        total = self.start_costs[path[0]]
        for step in range(1, self.length):
            total += self.cost(step, path[step - 1], path[step])
        return total


class TrellisBuilder:
    """Builds a :class:`CostTrellis` for a sentence from a language model.

    The cost function is injected, so swapping the paper's equation (1) for
    the standard log-likelihood changes nothing else in the pipeline.
    """

    def __init__(
        self, model: LanguageModel, cost: CostFunction = paper_distance
    ) -> None:
        self._model = model
        self._cost = cost

    def build(self, words: Sequence[str]) -> CostTrellis:
        if not words:
            raise ValueError("cannot build a trellis for an empty sentence")

        model, cost = self._model, self._cost
        emissions = [model.emission(word) for word in words]

        start_costs = tuple(
            cost(emissions[0][tag], model.initial[tag])
            for tag in range(model.tag_count)
        )
        step_costs = tuple(
            tuple(
                tuple(
                    cost(emission[tag], model.transition(previous_tag)[tag])
                    for tag in range(model.tag_count)
                )
                for previous_tag in range(model.tag_count)
            )
            for emission in emissions[1:]
        )

        return CostTrellis(
            tags=model.tags, start_costs=start_costs, step_costs=step_costs
        )
