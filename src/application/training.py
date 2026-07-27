"""Estimating a :class:`~pos_aco.model.LanguageModel` from a tagged corpus.

Maximum likelihood counting, with two adjustments that matter on real data:

* transitions and the initial distribution get add-k smoothing, so a tag pair
  the corpus happens not to contain is merely unlikely rather than impossible;
* rare training words are folded into :data:`~pos_aco.model.UNKNOWN_WORD`,
  which is what lets the tagger handle words it has never seen.

Emissions are left unsmoothed: smoothing them would fill in every one of the
``vocabulary x tags`` cells, and the whole point of the sparse lexicon is that
almost all of them are empty.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from ..domain.corpus import TaggedSentence
from ..domain.model import UNKNOWN_WORD, LanguageModel
from ..domain.tokenizer import DefaultTokenizer, Tokenizer


class ModelEstimator:
    """Counts a corpus into probabilities.

    ``unknown_threshold`` is the count at or below which a training word is
    treated as unseen. The default of 1 uses the hapax legomena - words
    occurring exactly once - which is the usual stand-in for "a word you have
    never met", since half of any corpus's vocabulary looks like that.
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        unknown_threshold: int = 1,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if smoothing < 0.0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")
        if unknown_threshold < 0:
            raise ValueError(
                f"unknown_threshold must be non-negative, got {unknown_threshold}"
            )
        self._smoothing = smoothing
        self._unknown_threshold = unknown_threshold
        # Must be the same tokenizer the tagger will later use, or the lexicon
        # keys will not match the words looked up against them.
        self._tokenizer = tokenizer or DefaultTokenizer()

    def estimate(self, sentences: Iterable[TaggedSentence]) -> LanguageModel:
        prepared = self._prepare(sentences)
        if not prepared:
            raise ValueError("cannot estimate a model from an empty corpus")

        tags = tuple(sorted({tag for sentence in prepared for _, tag in sentence}))
        return LanguageModel(
            tags=tags,
            emissions=self._emissions(prepared, tags),
            transitions=self._transitions(prepared, tags),
            initial=self._initial(prepared, tags),
        )

    def _prepare(self, sentences: Iterable[TaggedSentence]) -> list[list[tuple[str, str]]]:
        """Normalize words, drop bare punctuation, and mask rare words."""
        normalized = []
        for sentence in sentences:
            words = [(self._tokenizer.normalize(word), tag) for word, tag in sentence]
            kept = [(word, tag) for word, tag in words if word]
            if kept:
                normalized.append(kept)

        counts = Counter(word for sentence in normalized for word, _ in sentence)
        return [
            [
                (word if counts[word] > self._unknown_threshold else UNKNOWN_WORD, tag)
                for word, tag in sentence
            ]
            for sentence in normalized
        ]

    def _emissions(
        self, sentences: Sequence[Sequence[tuple[str, str]]], tags: tuple[str, ...]
    ) -> dict[str, tuple[float, ...]]:
        """P(word | tag), stored densely per word but only for seen words."""
        pairs: Counter[tuple[str, str]] = Counter()
        per_tag: Counter[str] = Counter()
        for sentence in sentences:
            for word, tag in sentence:
                pairs[(word, tag)] += 1
                per_tag[tag] += 1

        index = {tag: position for position, tag in enumerate(tags)}
        rows: dict[str, list[float]] = {}
        for (word, tag), count in pairs.items():
            row = rows.setdefault(word, [0.0] * len(tags))
            row[index[tag]] = count / per_tag[tag]
        return {word: tuple(row) for word, row in rows.items()}

    def _transitions(
        self, sentences: Sequence[Sequence[tuple[str, str]]], tags: tuple[str, ...]
    ) -> tuple[tuple[float, ...], ...]:
        """P(tag | previous tag), add-k smoothed."""
        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for sentence in sentences:
            for (_, previous), (_, tag) in zip(sentence, sentence[1:]):
                counts[previous][tag] += 1

        return tuple(
            self._smoothed(counts[previous], tags) for previous in tags
        )

    def _initial(
        self, sentences: Sequence[Sequence[tuple[str, str]]], tags: tuple[str, ...]
    ) -> tuple[float, ...]:
        """P(tag) for the first word of a sentence, add-k smoothed."""
        counts = Counter(sentence[0][1] for sentence in sentences)
        return self._smoothed(counts, tags)

    def _smoothed(self, counts: Counter[str], tags: tuple[str, ...]) -> tuple[float, ...]:
        total = sum(counts.values()) + self._smoothing * len(tags)
        if total == 0:
            # An unsmoothed tag that never starts a sentence or never precedes
            # anything: fall back to uniform rather than dividing by zero.
            return tuple(1.0 / len(tags) for _ in tags)
        return tuple((counts[tag] + self._smoothing) / total for tag in tags)
