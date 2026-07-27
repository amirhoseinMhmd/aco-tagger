"""Part-of-speech tagging with ant colony optimisation.

Reference implementation of ACO-tagger (arXiv:2303.16760).

The package is layered, and a layer may only import the ones above it:

===============  ===========================================================
``domain``       the model, the cost trellis, the solver interface, text
                 normalisation. Pure; standard library only.
``search``       exact (Viterbi) and approximate (ant colony) path finding.
``application``  the use cases - tag, train, evaluate - plus the ports it
                 needs the outside world to satisfy.
``adapters``     CSV persistence and corpus readers, implementing those ports.
===============  ===========================================================

``cli`` and ``train`` sit on top and wire the layers together.
``tests/test_architecture.py`` fails if that direction is ever violated.

Tagging:  text -> words (domain.tokenizer) -> costs (domain.trellis)
          -> cheapest path (search) -> tags (application.tagger).
Training: a corpus (adapters.corpora) -> counts (application.training)
          -> CSVs (adapters.csv_model).
"""

from .adapters import (
    CorpusError,
    CsvLanguageModelLoader,
    CsvLanguageModelWriter,
    load_treebank,
    read_conllu,
    read_tagged_file,
)
from .application import (
    Accuracy,
    LanguageModelLoader,
    LanguageModelWriter,
    ModelEstimator,
    PosTagger,
    Tagging,
    evaluate,
)
from .domain import (
    DEFAULT_TOKENIZER,
    UNKNOWN_WORD,
    CostFunction,
    CostTrellis,
    DefaultTokenizer,
    LanguageModel,
    ModelError,
    PathSolver,
    PersianTokenizer,
    Solution,
    TaggedSentence,
    Tokenizer,
    TrellisBuilder,
    TurkishTokenizer,
    UnknownWordError,
    get_tokenizer,
    negative_log_likelihood,
    normalize,
    paper_distance,
    split,
    tokenize,
)
from .search import (
    PAPER_DEPOSIT_FACTOR,
    PAPER_PARAMETERS,
    AntColonySolver,
    AntCycle,
    AntDensity,
    AntQuantity,
    ColonyParameters,
    DepositStrategy,
    ViterbiSolver,
)

__all__ = [
    "DEFAULT_TOKENIZER",
    "PAPER_DEPOSIT_FACTOR",
    "PAPER_PARAMETERS",
    "UNKNOWN_WORD",
    "Accuracy",
    "AntColonySolver",
    "AntCycle",
    "AntDensity",
    "AntQuantity",
    "ColonyParameters",
    "CorpusError",
    "CostFunction",
    "CostTrellis",
    "CsvLanguageModelLoader",
    "CsvLanguageModelWriter",
    "DefaultTokenizer",
    "DepositStrategy",
    "LanguageModel",
    "LanguageModelLoader",
    "LanguageModelWriter",
    "ModelError",
    "ModelEstimator",
    "PathSolver",
    "PersianTokenizer",
    "PosTagger",
    "Solution",
    "TaggedSentence",
    "Tagging",
    "Tokenizer",
    "TrellisBuilder",
    "TurkishTokenizer",
    "UnknownWordError",
    "ViterbiSolver",
    "evaluate",
    "get_tokenizer",
    "load_treebank",
    "negative_log_likelihood",
    "normalize",
    "paper_distance",
    "read_conllu",
    "read_tagged_file",
    "split",
    "tokenize",
]
