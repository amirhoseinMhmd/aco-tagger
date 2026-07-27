"""Turning raw text into the word sequence the tagger consumes.

Everything else in this package treats words as opaque strings, so this module
is the *only* place a language assumption can live. Training and tagging must
use the same tokenizer or a model trained on one spelling can never be looked
up for another - which is why the choice is recorded alongside the model (see
:mod:`pos_aco.loading`).

Adding a language means adding a :class:`Tokenizer`, not touching the model,
the trellis, or the solvers.
"""

from __future__ import annotations

import unicodedata
from typing import Protocol

# Zero-width non-joiner, which appears *inside* Persian words (mi-rawad) and
# must survive tokenization.
ZWNJ = "‌"


class Tokenizer(Protocol):
    """Splits text into words and canonicalises a single word."""

    def normalize(self, word: str) -> str: ...

    def tokenize(self, text: str) -> tuple[str, ...]: ...


def _is_punctuation(character: str) -> bool:
    """True for any Unicode punctuation, not just the ASCII marks.

    This is what makes the Persian comma, the Arabic semicolon, the Devanagari
    danda and the CJK full stop behave like ``.`` and ``,`` do in English.
    """
    return unicodedata.category(character).startswith("P")


def strip_punctuation(word: str) -> str:
    """Remove leading and trailing punctuation, keeping the inside intact.

    Only the ends are trimmed, so ``dit-il``, ``don't`` and ``mi<ZWNJ>rawad``
    survive as single words.
    """
    start, end = 0, len(word)
    while start < end and _is_punctuation(word[start]):
        start += 1
    while end > start and _is_punctuation(word[end - 1]):
        end -= 1
    return word[start:end]


class DefaultTokenizer:
    """Language-neutral tokenizer: whitespace split, Unicode-aware cleanup.

    Suitable for any language that separates words with whitespace. Languages
    that do not - Chinese, Japanese, Thai, Khmer, Lao - need a segmenter in
    front of this; see the README.
    """

    def normalize(self, word: str) -> str:
        """The canonical lexicon form. An empty result means "not a word"."""
        return self._fold_case(self._fold_characters(strip_punctuation(word)))

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(word for word in map(self.normalize, text.split()) if word)

    def _fold_characters(self, word: str) -> str:
        """Hook for script-level spelling variation. Neutral by default."""
        return word

    def _fold_case(self, word: str) -> str:
        """Hook for case folding. ``casefold`` beats ``lower`` for matching."""
        return word.casefold()


# Arabic-script spellings that Persian text mixes freely. Left unfolded, the
# same word written two ways becomes two lexicon entries, quietly splitting
# counts and inflating the out-of-vocabulary rate. Unicode NFC does not merge
# these - they are distinct letters, not composition variants.
_PERSIAN_CHARACTERS = {
    "ك": "ک",  # Arabic kaf          -> Persian keheh
    "ي": "ی",  # Arabic yeh          -> Farsi yeh
    "ى": "ی",  # alef maksura        -> Farsi yeh
    "ة": "ه",  # teh marbuta         -> heh
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "آ": "ا",  # alef with madda
    "ـ": "",  # tatweel (kashida), pure decoration
}
# Arabic-Indic digits -> the Persian forms.
_PERSIAN_CHARACTERS.update(
    {chr(0x0660 + digit): chr(0x06F0 + digit) for digit in range(10)}
)
# Harakat (short-vowel diacritics), optional in writing and usually absent.
_PERSIAN_CHARACTERS.update({chr(mark): "" for mark in range(0x064B, 0x0653)})

_PERSIAN_TABLE = str.maketrans(_PERSIAN_CHARACTERS)


class PersianTokenizer(DefaultTokenizer):
    """Persian / Farsi, folding the Arabic-script spelling variants.

    Persian has no letter case, so folding case is a no-op; the work is all in
    unifying characters that look identical but differ in codepoint.
    """

    def _fold_characters(self, word: str) -> str:
        return word.translate(_PERSIAN_TABLE)


class TurkishTokenizer(DefaultTokenizer):
    """Turkish / Azerbaijani, where the dotted and dotless I are distinct.

    Python's case folding is locale-independent, so ``I`` becomes ``i`` when
    Turkish requires ``ı``. The two letters are separate phonemes, so getting
    this wrong merges unrelated words.
    """

    def _fold_case(self, word: str) -> str:
        return word.replace("I", "ı").replace("İ", "i").casefold()


TOKENIZERS: dict[str, type[DefaultTokenizer]] = {
    "default": DefaultTokenizer,
    "persian": PersianTokenizer,
    "turkish": TurkishTokenizer,
}

DEFAULT_TOKENIZER = "default"


def get_tokenizer(name: str = DEFAULT_TOKENIZER) -> Tokenizer:
    """Look a tokenizer up by name, as the CLIs and model metadata do."""
    try:
        return TOKENIZERS[name]()
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"unknown tokenizer {name!r}; known: {known}") from None


_DEFAULT = DefaultTokenizer()


def normalize(word: str) -> str:
    """Normalize with the default tokenizer."""
    return _DEFAULT.normalize(word)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize with the default tokenizer."""
    return _DEFAULT.tokenize(text)
