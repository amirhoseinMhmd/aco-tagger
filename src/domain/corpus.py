"""What a tagged corpus *is*, independent of where it comes from.

The type and the split live here rather than beside the file readers because
:mod:`pos_aco.application.training` and :mod:`pos_aco.application.evaluation`
need them without caring whether the sentences came from CoNLL-U, a plain
file, or a list built in a test.
"""

from __future__ import annotations

from typing import Sequence

# A sentence is a sequence of (word, tag) pairs. Anything that can produce
# this shape can train a model.
TaggedSentence = Sequence[tuple[str, str]]


def split(
    sentences: Sequence[TaggedSentence], holdout: float = 0.1
) -> tuple[list[TaggedSentence], list[TaggedSentence]]:
    """Split into (train, test) by position, keeping the split reproducible.

    The fraction is floored, so training always keeps at least one sentence
    for any ``holdout`` below 1.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {holdout}")
    cut = len(sentences) - int(len(sentences) * holdout)
    return list(sentences[:cut]), list(sentences[cut:])
