"""Fidelity to arXiv:2303.16760, "ACO-tagger".

The numbers asserted here are the paper's own: the distances worked through on
page 5, and the parameters of table 4. They are what pins the implementation
to the published method rather than to a plausible reimplementation of it.
"""

from __future__ import annotations

import io
import math
import random
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import product

from pos_aco.search.colony import (
    PAPER_DEPOSIT_FACTOR,
    PAPER_PARAMETERS,
    AntColonySolver,
)
from pos_aco.cli import main
from pos_aco.application.tagger import PosTagger
from pos_aco.domain.trellis import (
    TrellisBuilder,
    negative_log_likelihood,
    paper_distance,
)
from pos_aco.search.viterbi import ViterbiSolver

from .support import (
    DATA_DIR,
    PAPER_EMISSIONS,
    PAPER_INITIAL,
    PAPER_SENTENCE,
    PAPER_TAGS,
    PAPER_TRANSITIONS,
    PAPER_TRANSITIONS_AS_PRINTED,
    paper_model,
)

N, V, ADJ, ADV, DELM = range(5)


class PaperDistanceTest(unittest.TestCase):
    """Equation (1): D = emission ** log10(transition)."""

    def test_matches_the_first_worked_example(self):
        # D(0, emruz|N) = 0.1 ** log10(0.6) = 1.667
        self.assertAlmostEqual(paper_distance(0.1, 0.6), 1.667, places=3)

    def test_matches_the_second_worked_example(self):
        # D(0, emruz|ADV) = 0.9 ** log10(0.3) = 1.057
        self.assertAlmostEqual(paper_distance(0.9, 0.3), 1.057, places=3)

    def test_matches_the_third_worked_example(self):
        # D(emruz|N, hava|N) = 1.0 ** log10(0.6) = 1
        self.assertAlmostEqual(paper_distance(1.0, 0.6), 1.0, places=6)

    def test_a_zero_probability_is_unreachable(self):
        self.assertTrue(math.isinf(paper_distance(0.0, 0.6)))
        self.assertTrue(math.isinf(paper_distance(0.1, 0.0)))

    def test_never_costs_less_than_one(self):
        for emission in (0.01, 0.5, 1.0):
            for transition in (0.01, 0.5, 1.0):
                self.assertGreaterEqual(paper_distance(emission, transition), 1.0)

    def test_a_certain_emission_ignores_the_transition_entirely(self):
        """The degeneracy: x ** 0 == 1, so the transition model drops out.

        An impossible transition costs exactly as much as a certain one when
        the emission probability is 1.
        """
        self.assertEqual(paper_distance(1.0, 0.5), paper_distance(1.0, 1e-9))

    def test_differs_from_the_standard_hmm_cost(self):
        """The two are not rescalings of each other; they rank paths differently."""
        self.assertNotAlmostEqual(
            paper_distance(0.1, 0.6), negative_log_likelihood(0.1, 0.6)
        )


class PaperTrellisTest(unittest.TestCase):
    """The trellis of figure 2, built from tables 1-3."""

    def setUp(self):
        self.trellis = TrellisBuilder(paper_model(), paper_distance).build(
            PAPER_SENTENCE
        )

    def test_has_one_segment_per_word(self):
        self.assertEqual(self.trellis.length, 5)
        self.assertEqual(self.trellis.tag_count, 5)

    def test_reproduces_the_initial_edge_costs(self):
        # The five D(0, emruz|tag) values listed on page 5.
        self.assertAlmostEqual(self.trellis.cost(0, 0, N), 1.667, places=3)
        self.assertTrue(math.isinf(self.trellis.cost(0, 0, V)))
        self.assertTrue(math.isinf(self.trellis.cost(0, 0, ADJ)))
        self.assertAlmostEqual(self.trellis.cost(0, 0, ADV), 1.057, places=3)
        self.assertTrue(math.isinf(self.trellis.cost(0, 0, DELM)))

    def test_reproduces_the_second_segment_costs(self):
        # D(emruz|N, hava|tag): only N is reachable.
        self.assertAlmostEqual(self.trellis.cost(1, N, N), 1.0, places=6)
        for tag in (V, ADJ, ADV, DELM):
            self.assertTrue(math.isinf(self.trellis.cost(1, N, tag)))

    def test_tags_the_paper_sentence_correctly(self):
        solution = ViterbiSolver().solve(self.trellis)
        self.assertEqual(
            paper_model().names_of(solution.path), ("ADV", "N", "ADJ", "V", "DELM")
        )


class PaperParametersTest(unittest.TestCase):
    """Table 4."""

    def test_matches_the_published_values(self):
        self.assertEqual(PAPER_PARAMETERS.generations, 3)
        self.assertEqual(PAPER_PARAMETERS.ant_count, 20)
        self.assertEqual(PAPER_PARAMETERS.alpha, 0.9)
        self.assertEqual(PAPER_PARAMETERS.beta, 0.9)
        self.assertEqual(PAPER_PARAMETERS.evaporation, 0.95)
        self.assertEqual(PAPER_DEPOSIT_FACTOR, 10.0)

    def test_the_colony_reaches_the_same_answer_as_viterbi_here(self):
        tagger = PosTagger(
            paper_model(),
            AntColonySolver(PAPER_PARAMETERS, rng=random.Random(0)),
            paper_distance,
        )
        self.assertEqual(
            tagger.tag_words(PAPER_SENTENCE).tags,
            ("ADV", "N", "ADJ", "V", "DELM"),
        )


class PrintedTransitionTableTest(unittest.TestCase):
    """Table 2's orientation, recovered from the example sentence.

    Locks in the reason ``support.PAPER_TRANSITIONS`` transposes the printed
    table, so nobody "corrects" it back later.
    """

    def cost(self, path, transitions) -> float:
        total = paper_distance(
            PAPER_EMISSIONS[PAPER_SENTENCE[0]][path[0]], PAPER_INITIAL[path[0]]
        )
        for step in range(1, len(path)):
            total += paper_distance(
                PAPER_EMISSIONS[PAPER_SENTENCE[step]][path[step]],
                transitions[path[step - 1]][path[step]],
            )
        return total

    def all_paths(self):
        return product(range(len(PAPER_TAGS)), repeat=len(PAPER_SENTENCE))

    def test_reading_rows_as_the_previous_tag_makes_every_tagging_impossible(self):
        finite = [
            path
            for path in self.all_paths()
            if math.isfinite(self.cost(path, PAPER_TRANSITIONS_AS_PRINTED))
        ]
        self.assertEqual(finite, [])

    def test_the_two_zeros_that_block_the_sentence(self):
        as_printed = PAPER_TRANSITIONS_AS_PRINTED
        self.assertEqual(as_printed[ADJ][V], 0.0)  # "barfi/ADJ ast/V" blocked
        self.assertEqual(as_printed[V][DELM], 0.0)  # "ast/V ./DELM" blocked

    def test_transposing_makes_the_intended_tagging_the_cheapest(self):
        finite = [
            path
            for path in self.all_paths()
            if math.isfinite(self.cost(path, PAPER_TRANSITIONS))
        ]
        self.assertEqual(len(finite), 4)

        best = min(finite, key=lambda path: self.cost(path, PAPER_TRANSITIONS))
        self.assertEqual(best, (ADV, N, ADJ, V, DELM))
        self.assertAlmostEqual(self.cost(best, PAPER_TRANSITIONS), 5.1261, places=4)


class ExactSolverDominanceTest(unittest.TestCase):
    """Why 'ACO beats Viterbi' cannot be a statement about search quality.

    Viterbi is exact for any additive edge cost, equation (1) included, so on
    a shared cost function it can never be beaten on cost. Any accuracy win
    for the colony therefore comes from the objective being misaligned with
    correctness, not from the colony searching better.
    """

    def test_viterbi_is_never_beaten_on_the_paper_cost(self):
        trellis = TrellisBuilder(paper_model(), paper_distance).build(PAPER_SENTENCE)
        exact = ViterbiSolver().solve(trellis).cost

        for seed in range(20):
            colony = AntColonySolver(PAPER_PARAMETERS, rng=random.Random(seed))
            self.assertGreaterEqual(colony.solve(trellis).cost + 1e-9, exact)


class PaperCommandLineTest(unittest.TestCase):
    def run_cli(self, *arguments) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["--data-dir", str(DATA_DIR), *arguments])
        return code, out.getvalue()

    def test_the_paper_cost_is_the_default(self):
        default = self.run_cli("john is dead", "--solver", "viterbi")
        explicit = self.run_cli("john is dead", "--solver", "viterbi", "--cost", "paper")
        self.assertEqual(default, explicit)

    def test_the_two_cost_functions_report_different_costs(self):
        _, paper = self.run_cli("john is dead", "--solver", "viterbi", "--cost", "paper")
        _, standard = self.run_cli(
            "john is dead", "--solver", "viterbi", "--cost", "loglikelihood"
        )
        self.assertNotEqual(paper, standard)

    def test_the_paper_preset_runs(self):
        code, out = self.run_cli("john is dead", "--preset", "paper", "--seed", "1")
        self.assertEqual(code, 0)
        self.assertIn("cost:", out)


if __name__ == "__main__":
    unittest.main()
