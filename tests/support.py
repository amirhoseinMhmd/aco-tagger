"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

from pos_aco.domain.model import LanguageModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "toy"
TREEBANK_DIR = ROOT / "data" / "treebank"

# The sample model, spelled out so tests do not depend on the CSV files.
TAGS = ("NOUN", "VERB", "ART")
EMISSIONS = {
    "john": (0.8, 0.1, 0.1),
    "is": (0.15, 0.15, 0.7),
    "dead": (0.05, 0.9, 0.05),
}
TRANSITIONS = (
    (0.6, 0.2, 0.2),
    (0.2, 0.1, 0.7),
    (0.5, 0.4, 0.1),
)
INITIAL = (0.6, 0.1, 0.3)

# Verified by exhaustive enumeration of all 27 taggings.
OPTIMAL_PATH = (0, 2, 1)  # john/NOUN is/ART dead/VERB
OPTIMAL_COST = 1.6163


def sample_model(**overrides) -> LanguageModel:
    fields = {
        "tags": TAGS,
        "emissions": EMISSIONS,
        "transitions": TRANSITIONS,
        "initial": INITIAL,
    }
    fields.update(overrides)
    return LanguageModel(**fields)


# ---------------------------------------------------------------------------
# The worked example from the ACO-tagger paper (arXiv:2303.16760), tables 1-3.
# The sentence is "emruz hava barfi ast ." - "today the weather is snowy".
# Transcribed from the Persian so the tests stay readable.
PAPER_TAGS = ("N", "V", "ADJ", "ADV", "DELM")

PAPER_EMISSIONS = {  # table 1, transposed to word -> probability per tag
    "emruz": (0.1, 0.0, 0.0, 0.9, 0.0),
    "hava": (1.0, 0.0, 0.0, 0.0, 0.0),
    "barfi": (0.2, 0.0, 0.8, 0.0, 0.0),
    "ast": (0.0, 1.0, 0.0, 0.0, 0.0),
    ".": (0.0, 0.0, 0.0, 0.0, 1.0),
}

PAPER_TRANSITIONS_AS_PRINTED = (  # table 2, exactly as laid out in the paper
    (0.6, 0.05, 0.2, 0.05, 0.2),
    (0.7, 0.1, 0.2, 0.0, 0.0),
    (0.5, 0.0, 0.1, 0.15, 0.25),
    (0.35, 0.05, 0.3, 0.1, 0.2),
    (0.2, 0.7, 0.05, 0.05, 0.0),
)

# Table 2's orientation is not determined by the paper. The worked examples
# cannot pin it down: every second-segment case is either emission = 1 (giving
# D = 1 whatever the transition) or emission = 0 (giving inf whatever it is).
#
# It is recoverable from the example sentence instead. Read with rows as the
# previous tag, ADJ->V and V->DELM are both zero, and the sentence needs both,
# so all 3125 taggings cost inf. Transposed, exactly 4 are finite and the
# cheapest is the intended "today the weather is snowy". Columns are therefore
# the previous tag, and this codebase's row-is-previous convention needs the
# transpose. See test_paper.PrintedTransitionTableTest.
PAPER_TRANSITIONS = tuple(zip(*PAPER_TRANSITIONS_AS_PRINTED))

PAPER_INITIAL = (0.6, 0.01, 0.04, 0.3, 0.05)  # table 3

PAPER_SENTENCE = ["emruz", "hava", "barfi", "ast", "."]


def paper_model() -> LanguageModel:
    return LanguageModel(
        tags=PAPER_TAGS,
        emissions=PAPER_EMISSIONS,
        transitions=PAPER_TRANSITIONS,
        initial=PAPER_INITIAL,
    )
