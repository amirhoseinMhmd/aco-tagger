"""Measuring a tagger against a held-out corpus.

Accuracy is split by whether the model had ever seen the word, because those
two numbers say very different things: the known-word figure measures the
model, the unknown-word figure measures how well transitions alone cope.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..domain.corpus import TaggedSentence
from ..domain.model import LanguageModel
from .tagger import PosTagger
from ..domain.tokenizer import DefaultTokenizer, Tokenizer


@dataclass(frozen=True)
class Accuracy:
    """Correct-token counts over a held-out corpus."""

    correct: int
    total: int
    correct_known: int
    total_known: int
    seconds: float

    @property
    def overall(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def known(self) -> float:
        return self.correct_known / self.total_known if self.total_known else 0.0

    @property
    def total_unknown(self) -> int:
        return self.total - self.total_known

    @property
    def unknown(self) -> float:
        correct = self.correct - self.correct_known
        return correct / self.total_unknown if self.total_unknown else 0.0

    def __str__(self) -> str:
        return (
            f"{self.overall:6.2%} overall ({self.correct}/{self.total} tokens)  "
            f"{self.known:6.2%} known ({self.total_known})  "
            f"{self.unknown:6.2%} unknown ({self.total_unknown})  "
            f"{self.seconds:.1f}s"
        )


def evaluate(
    tagger: PosTagger,
    model: LanguageModel,
    sentences: list[TaggedSentence],
    limit: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> Accuracy:
    """Tag each sentence and compare against its gold tags.

    ``limit`` caps how many sentences are scored, which keeps an ant colony
    run over a large corpus to a sensible wall time.
    """
    if limit is not None:
        sentences = sentences[:limit]
    tokenizer = tokenizer or DefaultTokenizer()

    correct = total = correct_known = total_known = 0
    started = time.perf_counter()

    for sentence in sentences:
        gold = [(tokenizer.normalize(word), tag) for word, tag in sentence]
        gold = [(word, tag) for word, tag in gold if word]
        if not gold:
            continue

        predicted = tagger.tag_words([word for word, _ in gold])
        for (word, expected), actual in zip(gold, predicted.tags):
            hit = expected == actual
            seen = word in model.emissions

            total += 1
            correct += hit
            total_known += seen
            correct_known += hit and seen

    return Accuracy(
        correct=correct,
        total=total,
        correct_known=correct_known,
        total_known=total_known,
        seconds=time.perf_counter() - started,
    )
