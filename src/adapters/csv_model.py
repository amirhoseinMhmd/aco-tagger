"""CSV persistence for a :class:`~pos_aco.domain.model.LanguageModel`.

The adapter behind :class:`~pos_aco.application.ports.LanguageModelLoader` and
:class:`~pos_aco.application.ports.LanguageModelWriter`: a different backing
store only needs another implementation of those two protocols.

The lexicon is stored in long form - one ``word,tag,probability`` row per
non-zero entry - because a real corpus fills only a few percent of the
``word x tag`` grid. ``initial.csv`` defines the canonical tag order, and the
other two files are validated against it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ..domain.model import LanguageModel, ModelError, Probabilities
from ..domain.tokenizer import DEFAULT_TOKENIZER

LEXICON_FILE = "lexicon.csv"
TRANSITIONS_FILE = "transitions.csv"
INITIAL_FILE = "initial.csv"
METADATA_FILE = "metadata.csv"

LEXICON_HEADER = ("word", "tag", "probability")
METADATA_HEADER = ("key", "value")

TOKENIZER_KEY = "tokenizer"



class CsvLanguageModelLoader:
    """Loads a model from three CSV files in one directory.

    ``initial.csv``     one column per tag, a single row - the tag order here
                        is authoritative.
    ``transitions.csv`` the same columns, one row per previous tag.
    ``lexicon.csv``     ``word,tag,probability``, non-zero entries only.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    def load(self) -> LanguageModel:
        tags, initial = self._read_initial()
        transitions = self._read_transitions(tags)
        emissions = self._read_lexicon(tags)

        return LanguageModel(
            tags=tags,
            emissions=emissions,
            transitions=transitions,
            initial=initial,
        )

    def _read_initial(self) -> tuple[tuple[str, ...], Probabilities]:
        path = self._path(INITIAL_FILE)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            tags = tuple(self._header(reader, path))
            rows = list(self._rows(reader, path, width=len(tags)))

        if len(rows) != 1:
            raise ModelError(f"{path}: expected a single row, got {len(rows)}")
        return tags, self._to_probabilities(rows[0], path)

    def _read_transitions(self, tags: tuple[str, ...]) -> tuple[Probabilities, ...]:
        path = self._path(TRANSITIONS_FILE)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(self._header(reader, path))
            if header != tags:
                raise ModelError(
                    f"{path}: tag columns {header} do not match {INITIAL_FILE}'s {tags}"
                )
            return tuple(
                self._to_probabilities(row, path)
                for row in self._rows(reader, path, width=len(tags))
            )

    def _read_lexicon(self, tags: tuple[str, ...]) -> dict[str, Probabilities]:
        path = self._path(LEXICON_FILE)
        index = {tag: position for position, tag in enumerate(tags)}
        rows: dict[str, list[float]] = {}

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(self._header(reader, path))
            if header != LEXICON_HEADER:
                raise ModelError(
                    f"{path}: expected columns {LEXICON_HEADER}, got {header}"
                )

            for word, tag, probability in self._rows(reader, path, width=3):
                if tag not in index:
                    raise ModelError(f"{path}: unknown tag {tag!r}")
                row = rows.setdefault(word, [0.0] * len(tags))
                row[index[tag]] = self._to_float(probability, path)

        if not rows:
            raise ModelError(f"{path}: no lexicon entries")
        return {word: tuple(row) for word, row in rows.items()}

    def _header(self, reader: Iterable[list[str]], path: Path) -> list[str]:
        for row in reader:
            return [column.strip() for column in row]
        raise ModelError(f"{path}: file is empty")

    def _rows(
        self, reader: Iterable[list[str]], path: Path, width: int
    ) -> Iterator[list[str]]:
        for number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != width:
                raise ModelError(
                    f"{path}: line {number} has {len(row)} fields, expected {width}"
                )
            yield row

    def _to_probabilities(self, values: Iterable[str], path: Path) -> Probabilities:
        return tuple(self._to_float(value, path) for value in values)

    def _to_float(self, value: str, path: Path) -> float:
        try:
            return float(value)
        except ValueError as error:
            raise ModelError(f"{path}: {error}") from None

    def metadata(self) -> dict[str, str]:
        """Whatever ``metadata.csv`` records, or ``{}`` if there is none.

        Optional so that model directories written before metadata existed
        still load.
        """
        path = self._directory / METADATA_FILE
        if not path.is_file():
            return {}

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(self._header(reader, path))
            if header != METADATA_HEADER:
                raise ModelError(
                    f"{path}: expected columns {METADATA_HEADER}, got {header}"
                )
            return {
                key: value for key, value in self._rows(reader, path, width=2)
            }

    def tokenizer_name(self, default: str = DEFAULT_TOKENIZER) -> str:
        """The tokenizer this model was trained with.

        Reading it back is what stops a model trained with, say, the Persian
        tokenizer from being queried with the default one - the lookups would
        silently miss and every word would come back unknown.
        """
        return self.metadata().get(TOKENIZER_KEY, default)

    def _path(self, filename: str) -> Path:
        path = self._directory / filename
        if not path.is_file():
            raise ModelError(f"missing model file: {path}")
        return path


class CsvLanguageModelWriter:
    """Writes a model back out in the format the loader expects.

    The inverse of :class:`CsvLanguageModelLoader`, so a trained model can be
    saved and reloaded without loss.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    def save(
        self,
        model: LanguageModel,
        tokenizer: str = DEFAULT_TOKENIZER,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._write(INITIAL_FILE, [model.tags, self._format(model.initial)])
        self._write(
            TRANSITIONS_FILE,
            [model.tags, *(self._format(row) for row in model.transitions)],
        )
        self._write(LEXICON_FILE, [LEXICON_HEADER, *self._lexicon_rows(model)])

        entries = {TOKENIZER_KEY: tokenizer, **(metadata or {})}
        self._write(
            METADATA_FILE,
            [METADATA_HEADER, *(sorted(entries.items()))],
        )

    def _lexicon_rows(self, model: LanguageModel) -> Iterator[list[str]]:
        """One row per non-zero entry, in a stable order for clean diffs."""
        for word in sorted(model.emissions):
            for tag, probability in zip(model.tags, model.emissions[word]):
                if probability > 0.0:
                    yield [word, tag, self._number(probability)]

    def _write(self, filename: str, rows: Iterable[Iterable[str]]) -> None:
        path = self._directory / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    def _format(self, row: Probabilities) -> list[str]:
        return [self._number(value) for value in row]

    def _number(self, value: float) -> str:
        """Enough precision to round-trip, without pages of trailing digits."""
        return f"{value:.10g}"
