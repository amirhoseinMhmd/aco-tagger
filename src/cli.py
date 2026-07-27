"""Command line front end.

Wiring only: it turns arguments into objects and prints the result. All the
decisions live in the modules it composes.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .search.colony import (
    PAPER_DEPOSIT_FACTOR,
    PAPER_PARAMETERS,
    AntColonySolver,
    ColonyParameters,
)
from .adapters.csv_model import CsvLanguageModelLoader
from .domain.model import ModelError, UnknownWordError
from .search.pheromone import AntCycle, AntDensity, AntQuantity, DepositStrategy
from .domain.solver import PathSolver
from .application.tagger import PosTagger
from .domain.tokenizer import TOKENIZERS, get_tokenizer
from .domain.trellis import CostFunction, negative_log_likelihood, paper_distance
from .search.viterbi import ViterbiSolver

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "treebank"
DEFAULT_TEXT = "All we have to decide is what to do with the time that is given us."

DEPOSIT_STRATEGIES = {
    "cycle": AntCycle,
    "density": AntDensity,
    "quantity": AntQuantity,
}

COST_FUNCTIONS: dict[str, CostFunction] = {
    "paper": paper_distance,
    "loglikelihood": negative_log_likelihood,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pos-aco",
        description="Part-of-speech tagging by ant colony optimisation.",
    )
    parser.add_argument(
        "text", nargs="?", default=DEFAULT_TEXT, help="sentence to tag"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory holding lexicon.csv, transitions.csv and initial.csv",
    )
    parser.add_argument(
        "--solver",
        choices=("aco", "viterbi"),
        default="aco",
        help="aco approximates; viterbi is exact and useful for comparison",
    )
    parser.add_argument(
        "--cost",
        choices=tuple(COST_FUNCTIONS),
        default="paper",
        help="'paper' is the ACO-tagger equation (1); 'loglikelihood' is the "
        "standard HMM cost, which is what Viterbi is exact for",
    )
    parser.add_argument(
        "--preset",
        choices=("paper",),
        help="use the paper's Table 4 colony parameters "
        "(overrides --ants/--generations/--alpha/--beta/--evaporation/--q)",
    )
    parser.add_argument(
        "--tokenizer",
        choices=tuple(sorted(TOKENIZERS)),
        help="override the tokenizer recorded in the model's metadata.csv",
    )
    parser.add_argument("--seed", type=int, help="seed the colony for repeatable runs")
    parser.add_argument("--ants", type=int, default=ColonyParameters.ant_count)
    parser.add_argument(
        "--generations", type=int, default=ColonyParameters.generations
    )
    parser.add_argument(
        "--alpha", type=float, default=ColonyParameters.alpha, help="pheromone weight"
    )
    parser.add_argument(
        "--beta", type=float, default=ColonyParameters.beta, help="cost weight"
    )
    parser.add_argument(
        "--evaporation", type=float, default=ColonyParameters.evaporation
    )
    parser.add_argument(
        "--deposit", choices=tuple(DEPOSIT_STRATEGIES), default="cycle"
    )
    parser.add_argument("--q", type=float, default=1.0, help="deposit scale factor")
    return parser


def build_solver(arguments: argparse.Namespace) -> PathSolver:
    if arguments.solver == "viterbi":
        return ViterbiSolver()

    if arguments.preset == "paper":
        parameters, q = PAPER_PARAMETERS, PAPER_DEPOSIT_FACTOR
    else:
        q = arguments.q
        parameters = ColonyParameters(
            ant_count=arguments.ants,
            generations=arguments.generations,
            alpha=arguments.alpha,
            beta=arguments.beta,
            evaporation=arguments.evaporation,
        )
    deposit: DepositStrategy = DEPOSIT_STRATEGIES[arguments.deposit](q)
    return AntColonySolver(
        parameters=parameters,
        deposit=deposit,
        rng=random.Random(arguments.seed),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    try:
        loader = CsvLanguageModelLoader(arguments.data_dir)
        model = loader.load()
        # Default to however the model was trained, so a Persian model is
        # queried with Persian normalisation without the caller knowing.
        tokenizer = get_tokenizer(arguments.tokenizer or loader.tokenizer_name())
        tagger = PosTagger(
            model,
            build_solver(arguments),
            COST_FUNCTIONS[arguments.cost],
            tokenizer,
        )
        tagging = tagger.tag(arguments.text)
    except (ModelError, UnknownWordError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(tagging)
    print(f"cost: {tagging.cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
