"""Tokenizing, the tagger facade, and the command line."""

from __future__ import annotations

import io
import random
import unittest
from contextlib import redirect_stderr, redirect_stdout

from pos_aco.cli import main
from pos_aco.search.colony import AntColonySolver, ColonyParameters
from pos_aco.application.tagger import PosTagger, Tagging
from pos_aco.domain.tokenizer import tokenize
from pos_aco.domain.trellis import negative_log_likelihood
from pos_aco.search.viterbi import ViterbiSolver

from .support import DATA_DIR, OPTIMAL_COST, sample_model


class TokenizeTest(unittest.TestCase):
    def test_splits_on_whitespace(self):
        self.assertEqual(tokenize("John is dead"), ("john", "is", "dead"))

    def test_strips_punctuation(self):
        self.assertEqual(tokenize("John is dead, he was;"),
                         ("john", "is", "dead", "he", "was"))

    def test_lowercases(self):
        self.assertEqual(tokenize("John IS Dead"), ("john", "is", "dead"))

    def test_collapses_extra_whitespace(self):
        self.assertEqual(tokenize("  john   is  "), ("john", "is"))

    def test_drops_tokens_that_are_only_punctuation(self):
        # Punctuation is recognised by Unicode category, so dashes go the same
        # way as full stops rather than surviving on an ASCII technicality.
        self.assertEqual(tokenize("john -- is"), ("john", "is"))
        self.assertEqual(tokenize("john . is"), ("john", "is"))

    def test_keeps_punctuation_inside_a_word(self):
        self.assertEqual(tokenize("dit-il don't"), ("dit-il", "don't"))

    def test_handles_empty_input(self):
        self.assertEqual(tokenize("   "), ())


class PosTaggerTest(unittest.TestCase):
    def setUp(self):
        self.tagger = PosTagger(sample_model(), ViterbiSolver(), negative_log_likelihood)

    def test_tags_a_sentence(self):
        tagging = self.tagger.tag("john is dead")
        self.assertEqual(tagging.words, ("john", "is", "dead"))
        self.assertEqual(tagging.tags, ("NOUN", "ART", "VERB"))
        self.assertAlmostEqual(tagging.cost, OPTIMAL_COST, places=4)

    def test_normalizes_text_before_tagging(self):
        self.assertEqual(self.tagger.tag("John is dead.").tags, ("NOUN", "ART", "VERB"))

    def test_keeps_both_tags_when_a_word_repeats(self):
        """The old dict-keyed output silently dropped one of these."""
        tagging = self.tagger.tag("john is is")
        self.assertEqual(len(tagging.words), 3)
        self.assertEqual(len(tagging.tags), 3)
        self.assertEqual(tagging.words, ("john", "is", "is"))

    def test_swapping_the_solver_does_not_change_the_interface(self):
        colony = PosTagger(
            sample_model(),
            AntColonySolver(ColonyParameters(), rng=random.Random(0)),
            negative_log_likelihood,
        )
        self.assertEqual(colony.tag("john is dead").tags, self.tagger.tag("john is dead").tags)

    def test_rejects_an_empty_sentence(self):
        with self.assertRaises(ValueError):
            self.tagger.tag("")


class TaggingTest(unittest.TestCase):
    def setUp(self):
        self.tagging = Tagging(words=("john", "is"), tags=("NOUN", "ART"), cost=1.0)

    def test_pairs_words_with_tags_in_order(self):
        self.assertEqual(self.tagging.pairs, (("john", "NOUN"), ("is", "ART")))

    def test_renders_as_word_slash_tag(self):
        self.assertEqual(str(self.tagging), "john/NOUN is/ART")


class CommandLineTest(unittest.TestCase):
    def run_cli(self, *arguments) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--data-dir", str(DATA_DIR), "--cost", "loglikelihood", *arguments])
        return code, out.getvalue(), err.getvalue()

    def test_tags_with_the_colony_by_default(self):
        code, out, _ = self.run_cli("john is dead", "--seed", "1")
        self.assertEqual(code, 0)
        self.assertIn("john/NOUN is/ART dead/VERB", out)
        self.assertIn("cost: 1.6163", out)

    def test_the_exact_solver_agrees_with_the_colony(self):
        _, colony, _ = self.run_cli("john is dead", "--seed", "1")
        _, exact, _ = self.run_cli("john is dead", "--solver", "viterbi")
        self.assertEqual(colony, exact)

    def test_reports_an_unknown_word_without_a_traceback(self):
        code, _, err = self.run_cli("john is unknownword")
        self.assertEqual(code, 2)
        self.assertIn("unknownword", err)

    def test_reports_a_missing_data_directory(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--data-dir", "/nonexistent", "john"])
        self.assertEqual(code, 2)
        self.assertIn("initial.csv", err.getvalue())

    def test_rejects_invalid_parameters(self):
        code, _, err = self.run_cli("john is dead", "--ants", "0")
        self.assertEqual(code, 2)
        self.assertIn("ant_count", err)

    def test_accepts_each_deposit_strategy(self):
        for strategy in ("cycle", "density", "quantity"):
            with self.subTest(strategy=strategy):
                code, out, _ = self.run_cli(
                    "john is dead", "--deposit", strategy, "--seed", "1"
                )
                self.assertEqual(code, 0)
                self.assertIn("cost:", out)


if __name__ == "__main__":
    unittest.main()
