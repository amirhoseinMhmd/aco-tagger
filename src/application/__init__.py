"""The use cases: tag a sentence, estimate a model, score a model.

Depends on :mod:`pos_aco.domain` and :mod:`pos_aco.search`. It declares what it
needs from the outside world as protocols in :mod:`~pos_aco.application.ports`
and never imports an adapter, so nothing here knows about CSV or file paths.
"""

from .evaluation import Accuracy, evaluate
from .ports import LanguageModelLoader, LanguageModelWriter
from .tagger import PosTagger, Tagging
from .training import ModelEstimator

__all__ = [
    "Accuracy",
    "LanguageModelLoader",
    "LanguageModelWriter",
    "ModelEstimator",
    "PosTagger",
    "Tagging",
    "evaluate",
]
