"""Cost construction: probabilities in, additive costs out."""

from __future__ import annotations

import math
import unittest

from pos_aco.domain.trellis import TrellisBuilder, negative_log_likelihood, to_cost

from .support import OPTIMAL_COST, OPTIMAL_PATH, sample_model


class ToCostTest(unittest.TestCase):
    def test_certainty_is_free(self):
        self.assertEqual(to_cost(1.0), 0.0)

    def test_impossibility_is_infinite(self):
        self.assertTrue(math.isinf(to_cost(0.0)))
        self.assertTrue(math.isinf(to_cost(-1.0)))

    def test_less_likely_costs_more(self):
        self.assertGreater(to_cost(0.1), to_cost(0.5))


class TrellisBuilderTest(unittest.TestCase):
    def setUp(self):
        # These expectations are the standard HMM cost, so opt into it explicitly.
        self.builder = TrellisBuilder(sample_model(), negative_log_likelihood)
        self.trellis = self.builder.build(["john", "is", "dead"])

    def test_reports_sentence_and_tag_dimensions(self):
        self.assertEqual(self.trellis.length, 3)
        self.assertEqual(self.trellis.tag_count, 3)

    def test_first_word_combines_emission_with_the_initial_distribution(self):
        # -log10(P(john|NOUN) * P(NOUN)) = -log10(0.8 * 0.6)
        self.assertAlmostEqual(self.trellis.cost(0, 0, 0), 0.3188, places=4)
        self.assertAlmostEqual(self.trellis.cost(0, 0, 1), 2.0, places=4)

    def test_first_word_ignores_its_nonexistent_predecessor(self):
        costs = {self.trellis.cost(0, previous, 0) for previous in range(3)}
        self.assertEqual(len(costs), 1)

    def test_later_words_combine_emission_with_the_transition(self):
        # -log10(P(is|ART) * P(ART|NOUN)) = -log10(0.7 * 0.2)
        self.assertAlmostEqual(self.trellis.cost(1, 0, 2), 0.8539, places=4)
        # -log10(P(dead|VERB) * P(VERB|ART)) = -log10(0.9 * 0.4)
        self.assertAlmostEqual(self.trellis.cost(2, 2, 1), 0.4437, places=4)

    def test_totals_a_path_by_summing_its_edges(self):
        self.assertAlmostEqual(
            self.trellis.total_cost(OPTIMAL_PATH), OPTIMAL_COST, places=4
        )

    def test_rejects_a_path_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            self.trellis.total_cost((0, 1))

    def test_rejects_an_empty_sentence(self):
        with self.assertRaises(ValueError):
            self.builder.build([])

    def test_handles_a_one_word_sentence(self):
        trellis = self.builder.build(["john"])
        self.assertEqual(trellis.length, 1)
        self.assertAlmostEqual(trellis.total_cost((0,)), 0.3188, places=4)


if __name__ == "__main__":
    unittest.main()
