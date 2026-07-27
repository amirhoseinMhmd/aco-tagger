"""Command line for estimating a model from a tagged corpus.

    python -m pos_aco.train --corpus treebank --out data/treebank

Wiring only, like :mod:`pos_aco.cli`: it reads a corpus, hands it to the
estimator, writes the CSVs, and optionally scores the result.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .search.colony import AntColonySolver, ColonyParameters
from .adapters.corpora import (
    CorpusError,
    load_treebank,
    read_conllu,
    read_tagged_file,
)
from .domain.corpus import split
from .application.evaluation import evaluate
from .adapters.csv_model import CsvLanguageModelWriter
from .domain.model import UNKNOWN_WORD, ModelError
from .application.tagger import PosTagger
from .application.training import ModelEstimator
from .domain.tokenizer import TOKENIZERS, get_tokenizer
from .domain.trellis import negative_log_likelihood, paper_distance
from .search.viterbi import ViterbiSolver

DEFAULT_OUTPUT = Path("data/treebank")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pos-aco-train",
        description="Estimate a tagging model from a tagged corpus.",
    )
    parser.add_argument(
        "--corpus",
        default="treebank",
        help="'treebank' for the NLTK sample, or a path to a corpus file",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "tagged", "conllu"),
        default="auto",
        help="corpus format; 'auto' picks conllu for a .conllu suffix",
    )
    parser.add_argument(
        "--tagset",
        choices=("upos", "xpos"),
        default="upos",
        help="which CoNLL-U tag column to use",
    )
    parser.add_argument(
        "--tokenizer",
        choices=tuple(sorted(TOKENIZERS)),
        default="default",
        help="language-specific normalisation; recorded in the model so "
        "tagging reuses it automatically",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="directory to write the CSVs to"
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.1,
        help="fraction of sentences reserved for evaluation (0 to train on all)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=1.0, help="add-k for transitions and initial"
    )
    parser.add_argument(
        "--unknown-threshold",
        type=int,
        default=1,
        help=f"training words seen this often or less become {UNKNOWN_WORD}",
    )
    parser.add_argument(
        "--evaluate",
        choices=("none", "viterbi", "aco", "both"),
        default="viterbi",
        help="score the trained model on the held-out sentences",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=200,
        help="how many held-out sentences to score (the colony is slow)",
    )
    parser.add_argument(
        "--cost",
        choices=("paper", "loglikelihood"),
        default="paper",
        help="edge cost used when evaluating: the paper's equation (1) or the "
        "standard HMM log-likelihood",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the colony")
    return parser


def read_corpus(name: str, corpus_format: str = "auto", tagset: str = "upos"):
    if name == "treebank":
        return load_treebank()
    if corpus_format == "conllu" or (
        corpus_format == "auto" and name.endswith(".conllu")
    ):
        return read_conllu(name, tagset)
    return read_tagged_file(name)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    try:
        tokenizer = get_tokenizer(arguments.tokenizer)
        sentences = read_corpus(
            arguments.corpus, arguments.format, arguments.tagset
        )
        training, held_out = split(sentences, arguments.holdout)
        model = ModelEstimator(
            smoothing=arguments.smoothing,
            unknown_threshold=arguments.unknown_threshold,
            tokenizer=tokenizer,
        ).estimate(training)
        CsvLanguageModelWriter(arguments.out).save(model, arguments.tokenizer)
    except (CorpusError, ModelError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    _report_model(arguments, sentences, training, held_out, model)
    _report_accuracy(arguments, held_out, model, tokenizer)
    return 0


def _report_model(arguments, sentences, training, held_out, model) -> None:
    entries = sum(
        1 for row in model.emissions.values() for value in row if value > 0.0
    )
    print(f"corpus      {arguments.corpus}: {len(sentences)} sentences")
    print(f"split       {len(training)} training / {len(held_out)} held out")
    print(f"tags        {len(model.tags)}")
    print(f"vocabulary  {len(model.emissions)} words, {entries} non-zero entries")
    print(f"unknowns    {'modelled' if model.handles_unknown_words else 'not modelled'}")
    print(f"tokenizer   {arguments.tokenizer}")
    print(f"written to  {arguments.out}")


def _report_accuracy(arguments, held_out, model, tokenizer) -> None:
    if arguments.evaluate == "none" or not held_out:
        return

    solvers = {
        "viterbi": lambda: ViterbiSolver(),
        "aco": lambda: AntColonySolver(
            ColonyParameters(ant_count=10, generations=20),
            rng=random.Random(arguments.seed),
        ),
    }
    chosen = ("viterbi", "aco") if arguments.evaluate == "both" else (arguments.evaluate,)

    cost = paper_distance if arguments.cost == "paper" else negative_log_likelihood

    print()
    for name in chosen:
        tagger = PosTagger(model, solvers[name](), cost, tokenizer)
        score = evaluate(
            tagger, model, held_out, arguments.eval_limit, tokenizer
        )
        print(f"{name:8} {score}")


if __name__ == "__main__":
    raise SystemExit(main())
