"""The interfaces the application needs the outside world to satisfy.

Declared here, in the layer that *uses* them, and implemented out in
:mod:`pos_aco.adapters`. That inversion is what keeps the inner layers free of
any knowledge of CSV, file paths, or NLTK.
"""

from __future__ import annotations

from typing import Protocol

from ..domain.model import LanguageModel


class LanguageModelLoader(Protocol):
    """Anything that can produce a language model."""

    def load(self) -> LanguageModel: ...


class LanguageModelWriter(Protocol):
    """Anything that can persist a language model."""

    def save(self, model: LanguageModel) -> None: ...
