"""The statistical model behind the tagger.

A first-order hidden Markov model over part-of-speech tags: every word has an
emission probability per tag, every tag pair has a transition probability, and
the first word of a sentence draws its tag from an initial distribution.

This module knows nothing about files, costs or search - it is only the data
and its invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

Probabilities = tuple[float, ...]

# Lexicon entry standing in for every word the training corpus never saw.
# Models estimated from a real corpus provide it; the hand-written sample
# model does not, and raises instead.
UNKNOWN_WORD = "<UNK>"


class ModelError(Exception):
    """Raised when a language model is malformed."""


class UnknownWordError(KeyError):
    """Raised when a word has no entry in the lexicon."""

    def __init__(self, word: str) -> None:
        super().__init__(word)
        self.word = word

    def __str__(self) -> str:
        return f"no lexicon entry for {self.word!r}"


@dataclass(frozen=True)
class LanguageModel:
    """Emission, transition and initial distributions over a fixed tag set."""

    tags: tuple[str, ...]
    emissions: Mapping[str, Probabilities]
    transitions: tuple[Probabilities, ...]
    initial: Probabilities

    def __post_init__(self) -> None:
        if not self.tags:
            raise ModelError("the tag set is empty")
        if len(set(self.tags)) != len(self.tags):
            raise ModelError(f"duplicate tags: {self.tags}")
        if not self.emissions:
            raise ModelError("the lexicon is empty")

        self._check_row("initial distribution", self.initial)
        for word, row in self.emissions.items():
            self._check_row(f"emissions for {word!r}", row)
        if len(self.transitions) != self.tag_count:
            raise ModelError(
                f"expected {self.tag_count} transition rows, got {len(self.transitions)}"
            )
        for tag, row in zip(self.tags, self.transitions):
            self._check_row(f"transitions from {tag!r}", row)

    def _check_row(self, label: str, row: Probabilities) -> None:
        if len(row) != self.tag_count:
            raise ModelError(
                f"{label}: expected {self.tag_count} probabilities, got {len(row)}"
            )
        for value in row:
            if not 0.0 <= value <= 1.0:
                raise ModelError(f"{label}: {value} is not a probability")

    @property
    def tag_count(self) -> int:
        return len(self.tags)

    @property
    def handles_unknown_words(self) -> bool:
        """Whether the lexicon carries an :data:`UNKNOWN_WORD` entry."""
        return UNKNOWN_WORD in self.emissions

    def emission(self, word: str) -> Probabilities:
        """P(word | tag) for every tag, in tag-set order.

        Falls back to the :data:`UNKNOWN_WORD` distribution when the word is
        out of vocabulary, so a trained model can tag any sentence.
        """
        row = self.emissions.get(word)
        if row is None:
            row = self.emissions.get(UNKNOWN_WORD)
        if row is None:
            raise UnknownWordError(word)
        return row

    def transition(self, previous_tag: int) -> Probabilities:
        """P(tag | previous_tag) for every tag, in tag-set order."""
        return self.transitions[previous_tag]

    def name_of(self, tag: int) -> str:
        return self.tags[tag]

    def names_of(self, path: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.tags[tag] for tag in path)
