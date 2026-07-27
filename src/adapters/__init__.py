"""The outside world: files on disk and third-party corpora.

The only layer that knows about CSV columns, CoNLL-U fields or NLTK. It
implements the protocols declared in :mod:`pos_aco.application.ports`, so the
dependency points inward even though the data flows outward.
"""

from .corpora import CorpusError, load_treebank, read_conllu, read_tagged_file
from .csv_model import CsvLanguageModelLoader, CsvLanguageModelWriter

__all__ = [
    "CorpusError",
    "CsvLanguageModelLoader",
    "CsvLanguageModelWriter",
    "load_treebank",
    "read_conllu",
    "read_tagged_file",
]
