"""The user-facing entry point: text in, tagged words out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..domain.model import LanguageModel
from ..domain.solver import PathSolver
from ..domain.tokenizer import DefaultTokenizer, Tokenizer
from ..domain.trellis import CostFunction, TrellisBuilder, paper_distance


@dataclass(frozen=True)
class Tagging:
    """One word per tag, in sentence order, plus the cost of that assignment.

    Parallel tuples rather than a dict, so a word repeated in the sentence
    keeps both of its tags.
    """

    words: tuple[str, ...]
    tags: tuple[str, ...]
    cost: float

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.words, self.tags))

    def __str__(self) -> str:
        return " ".join(f"{word}/{tag}" for word, tag in self.pairs)


class PosTagger:
    """Composes a language model with a search strategy.

    The solver is injected, so the same tagger works with the exact Viterbi
    solver or the ant colony without knowing the difference.
    """

    def __init__(
        self,
        model: LanguageModel,
        solver: PathSolver,
        cost: CostFunction = paper_distance,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._model = model
        self._builder = TrellisBuilder(model, cost)
        self._solver = solver
        self._tokenizer = tokenizer or DefaultTokenizer()

    def tag(self, text: str) -> Tagging:
        """Tokenize ``text`` and tag it."""
        return self.tag_words(self._tokenizer.tokenize(text))

    def tag_words(self, words: Sequence[str]) -> Tagging:
        """Tag an already-tokenized sentence."""
        trellis = self._builder.build(words)
        solution = self._solver.solve(trellis)
        return Tagging(
            words=tuple(words),
            tags=self._model.names_of(solution.path),
            cost=solution.cost,
        )
