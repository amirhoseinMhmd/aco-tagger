"""Readers that turn corpus files into tagged sentences.

An adapter: this is the only place that knows about CoNLL-U columns, the
``word/TAG`` convention, or NLTK. The shape it produces -
:data:`~pos_aco.domain.corpus.TaggedSentence` - is defined in the domain.

NLTK is imported lazily: it is a training-time convenience, never needed to
load or run a tagger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..domain.corpus import TaggedSentence

SEPARATOR = "/"


class CorpusError(Exception):
    """Raised when a corpus cannot be read."""


def read_tagged_file(path: Path | str, separator: str = SEPARATOR) -> list[TaggedSentence]:
    """Read a ``word/TAG word/TAG`` file, one sentence per line.

    Splits each token on the *last* separator, so slashes inside a word (as in
    ``and/or``) survive.
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"no such corpus file: {path}")

    sentences = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            sentence = list(_parse_line(line, separator, path, number))
            if sentence:
                sentences.append(sentence)

    if not sentences:
        raise CorpusError(f"{path}: no sentences found")
    return sentences


def _parse_line(
    line: str, separator: str, path: Path, number: int
) -> Iterator[tuple[str, str]]:
    for token in line.split():
        word, found, tag = token.rpartition(separator)
        if not found or not word or not tag:
            raise CorpusError(
                f"{path}: line {number}: {token!r} is not word{separator}TAG"
            )
        yield word, tag


CONLLU_FORM = 1
CONLLU_UPOS = 3
CONLLU_XPOS = 4
CONLLU_FIELDS = 10

MISSING = "_"


def read_conllu(path: Path | str, tagset: str = "upos") -> list[TaggedSentence]:
    """Read a CoNLL-U file, the Universal Dependencies interchange format.

    This is the practical route to other languages: UD publishes treebanks in
    this one format for over a hundred of them, so a new language needs a
    download and a retrain rather than any code change.

    ``tagset`` picks the universal 17-tag ``upos`` column or the
    treebank-specific ``xpos`` column.
    """
    column = {"upos": CONLLU_UPOS, "xpos": CONLLU_XPOS}.get(tagset)
    if column is None:
        raise CorpusError(f"tagset must be 'upos' or 'xpos', got {tagset!r}")

    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"no such corpus file: {path}")

    sentences: list[TaggedSentence] = []
    current: list[tuple[str, str]] = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):  # sentence id, text, and other metadata
                continue

            fields = line.split("\t")
            if len(fields) != CONLLU_FIELDS:
                raise CorpusError(
                    f"{path}: line {number} has {len(fields)} columns, "
                    f"expected {CONLLU_FIELDS}"
                )
            if not _is_word_line(fields[0]):
                continue

            tag = fields[column]
            if tag != MISSING:
                current.append((fields[CONLLU_FORM], tag))

    if current:
        sentences.append(current)
    if not sentences:
        raise CorpusError(f"{path}: no sentences found")
    return sentences


def _is_word_line(identifier: str) -> bool:
    """Skip multiword ranges (``1-2``) and empty nodes (``1.1``).

    Both would double-count: the range repeats words its members already
    cover, and empty nodes are not surface words at all.
    """
    return "-" not in identifier and "." not in identifier


def load_treebank() -> list[TaggedSentence]:
    """The Penn Treebank sample bundled with NLTK, using its native tagset."""
    try:
        from nltk.corpus import treebank
    except ImportError as error:  # pragma: no cover - depends on environment
        raise CorpusError(
            "reading the treebank corpus needs nltk: pip install nltk"
        ) from error

    try:
        sentences = [list(sentence) for sentence in treebank.tagged_sents()]
    except LookupError as error:  # pragma: no cover - depends on environment
        raise CorpusError(
            "the treebank corpus is not downloaded: "
            "python -c \"import nltk; nltk.download('treebank')\""
        ) from error

    # The treebank marks elided constituents with -NONE-; they are not words.
    return [
        [(word, tag) for word, tag in sentence if tag != "-NONE-"]
        for sentence in sentences
        if sentence
    ]

