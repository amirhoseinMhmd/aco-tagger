"""Search: the exact solver, the colony, and the pheromone machinery."""

from __future__ import annotations

import random
import unittest
from itertools import product

from pos_aco.search.ant import Ant
from pos_aco.search.colony import AntColonySolver, ColonyParameters
from pos_aco.domain.model import LanguageModel
from pos_aco.search.pheromone import (
    AntCycle,
    AntDensity,
    AntQuantity,
    PheromoneTable,
    edges,
)
from pos_aco.domain.solver import Solution
from pos_aco.domain.trellis import CostTrellis, TrellisBuilder, negative_log_likelihood
from pos_aco.search.viterbi import ViterbiSolver

from .support import OPTIMAL_COST, OPTIMAL_PATH, TAGS, sample_model

SENTENCE = ["john", "is", "dead"]


def brute_force(trellis: CostTrellis) -> Solution:
    """The obviously-correct answer: enumerate every tagging."""
    paths = product(range(trellis.tag_count), repeat=trellis.length)
    best = min(paths, key=trellis.total_cost)
    return Solution(path=best, cost=trellis.total_cost(best))


def sample_trellis(words=SENTENCE) -> CostTrellis:
    """The toy trellis under the standard HMM cost, which OPTIMAL_COST assumes."""
    return TrellisBuilder(sample_model(), negative_log_likelihood).build(words)


def random_trellis(rng: random.Random, tag_count: int, length: int) -> CostTrellis:
    def row():
        return tuple(rng.uniform(0.01, 1.0) for _ in range(tag_count))

    words = [f"w{i}" for i in range(length)]
    model = LanguageModel(
        tags=tuple(f"T{i}" for i in range(tag_count)),
        emissions={word: row() for word in words},
        transitions=tuple(row() for _ in range(tag_count)),
        initial=row(),
    )
    return TrellisBuilder(model, negative_log_likelihood).build(words)


class ViterbiTest(unittest.TestCase):
    def test_finds_the_known_optimum(self):
        solution = ViterbiSolver().solve(sample_trellis())
        self.assertEqual(solution.path, OPTIMAL_PATH)
        self.assertAlmostEqual(solution.cost, OPTIMAL_COST, places=4)

    def test_word_order_changes_the_answer(self):
        solution = ViterbiSolver().solve(sample_trellis(["is", "john", "dead"]))
        self.assertEqual(solution.path, (2, 0, 1))
        self.assertAlmostEqual(solution.cost, 1.8204, places=4)

    def test_handles_a_one_word_sentence(self):
        solution = ViterbiSolver().solve(sample_trellis(["john"]))
        self.assertEqual(solution.path, (0,))

    def test_reported_cost_matches_the_path(self):
        trellis = sample_trellis()
        solution = ViterbiSolver().solve(trellis)
        self.assertAlmostEqual(solution.cost, trellis.total_cost(solution.path))

    def test_agrees_with_brute_force_on_random_models(self):
        rng = random.Random(1234)
        for trial in range(25):
            trellis = random_trellis(rng, tag_count=4, length=5)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(
                    ViterbiSolver().solve(trellis).cost, brute_force(trellis).cost
                )


class PheromoneTableTest(unittest.TestCase):
    def setUp(self):
        self.trellis = sample_trellis()
        self.table = PheromoneTable(self.trellis, initial_level=1.0)

    def test_starts_uniform(self):
        self.assertEqual(self.table.level(0, 0, 1), 1.0)
        self.assertEqual(self.table.level(2, 1, 0), 1.0)

    def test_deposit_accumulates_on_one_edge_only(self):
        self.table.deposit(1, 0, 2, 0.5)
        self.assertEqual(self.table.level(1, 0, 2), 1.5)
        self.assertEqual(self.table.level(1, 0, 1), 1.0)
        self.assertEqual(self.table.level(1, 1, 2), 1.0)

    def test_first_step_shares_one_row_across_predecessors(self):
        self.table.deposit(0, 0, 1, 1.0)
        self.assertEqual(self.table.level(0, 2, 1), 2.0)

    def test_evaporation_scales_every_trail(self):
        self.table.deposit(1, 0, 2, 1.0)
        self.table.evaporate(0.25)
        self.assertAlmostEqual(self.table.level(1, 0, 2), 1.5)
        self.assertAlmostEqual(self.table.level(0, 0, 0), 0.75)

    def test_rejects_an_evaporation_rate_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            self.table.evaporate(1.5)

    def test_rejects_a_non_positive_initial_level(self):
        with self.assertRaises(ValueError):
            PheromoneTable(self.trellis, initial_level=0.0)


class DepositStrategyTest(unittest.TestCase):
    def setUp(self):
        self.trellis = sample_trellis()
        self.table = PheromoneTable(self.trellis, initial_level=0.0001)
        self.solution = Solution(path=OPTIMAL_PATH, cost=OPTIMAL_COST)

    def test_edges_walks_the_whole_path(self):
        self.assertEqual(
            list(edges(self.solution)), [(0, 0, 0), (1, 0, 2), (2, 2, 1)]
        )

    def test_ant_cycle_rewards_every_edge_by_path_quality(self):
        AntCycle(q=2.0).deposit(self.table, self.trellis, self.solution)
        expected = 2.0 / OPTIMAL_COST
        for step, previous, tag in edges(self.solution):
            self.assertAlmostEqual(
                self.table.level(step, previous, tag) - 0.0001, expected
            )

    def test_ant_density_rewards_every_edge_equally(self):
        AntDensity(q=3.0).deposit(self.table, self.trellis, self.solution)
        self.assertAlmostEqual(self.table.level(1, 0, 2) - 0.0001, 3.0)

    def test_ant_quantity_rewards_cheap_edges_more(self):
        AntQuantity(q=1.0).deposit(self.table, self.trellis, self.solution)
        on_cheaper_edge = self.table.level(0, 0, 0)  # cost 0.3188
        on_dearer_edge = self.table.level(2, 2, 1)  # cost 0.4437
        self.assertGreater(on_cheaper_edge, on_dearer_edge)

    def test_no_strategy_touches_unused_edges(self):
        AntCycle().deposit(self.table, self.trellis, self.solution)
        self.assertEqual(self.table.level(1, 0, 1), 0.0001)


class AntTest(unittest.TestCase):
    def test_produces_a_full_path_with_a_matching_cost(self):
        trellis = sample_trellis()
        table = PheromoneTable(trellis)
        solution = Ant(trellis, table, 1.0, 2.0, random.Random(0)).walk()

        self.assertEqual(len(solution.path), trellis.length)
        self.assertAlmostEqual(solution.cost, trellis.total_cost(solution.path))

    def test_never_takes_an_impossible_edge(self):
        # ART becomes impossible for the first word.
        model = sample_model(initial=(0.7, 0.3, 0.0))
        trellis = TrellisBuilder(model, negative_log_likelihood).build(SENTENCE)
        table = PheromoneTable(trellis)
        rng = random.Random(5)

        for _ in range(100):
            self.assertNotEqual(Ant(trellis, table, 1.0, 2.0, rng).walk().path[0], 2)

    def test_falls_back_to_a_uniform_choice_when_all_edges_are_impossible(self):
        model = sample_model(initial=(0.0, 0.0, 0.0))
        trellis = TrellisBuilder(model, negative_log_likelihood).build(["john"])
        table = PheromoneTable(trellis)
        rng = random.Random(3)

        chosen = {Ant(trellis, table, 1.0, 2.0, rng).walk().path[0] for _ in range(50)}
        self.assertEqual(chosen, {0, 1, 2})


class ColonyParametersTest(unittest.TestCase):
    def test_rejects_a_colony_with_no_ants(self):
        with self.assertRaises(ValueError):
            ColonyParameters(ant_count=0)

    def test_rejects_a_run_with_no_generations(self):
        with self.assertRaises(ValueError):
            ColonyParameters(generations=0)

    def test_rejects_an_evaporation_rate_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            ColonyParameters(evaporation=-0.1)

    def test_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            ColonyParameters(alpha=-1.0)


class AntColonySolverTest(unittest.TestCase):
    def solver(self, seed=0, **parameters) -> AntColonySolver:
        return AntColonySolver(
            parameters=ColonyParameters(**parameters), rng=random.Random(seed)
        )

    def test_finds_the_known_optimum(self):
        solution = self.solver().solve(sample_trellis())
        self.assertEqual(solution.path, OPTIMAL_PATH)
        self.assertAlmostEqual(solution.cost, OPTIMAL_COST, places=4)

    def test_is_reproducible_for_a_given_seed(self):
        trellis = sample_trellis()
        first = self.solver(seed=42).solve(trellis)
        second = self.solver(seed=42).solve(trellis)
        self.assertEqual(first, second)

    def test_reported_cost_matches_the_path(self):
        trellis = sample_trellis()
        solution = self.solver(seed=7).solve(trellis)
        self.assertAlmostEqual(solution.cost, trellis.total_cost(solution.path))

    def test_never_beats_the_exact_solver(self):
        rng = random.Random(99)
        for trial in range(10):
            trellis = random_trellis(rng, tag_count=4, length=5)
            optimum = ViterbiSolver().solve(trellis).cost
            found = self.solver(seed=trial).solve(trellis).cost
            with self.subTest(trial=trial):
                self.assertGreaterEqual(found + 1e-9, optimum)

    def test_works_with_every_deposit_strategy(self):
        trellis = sample_trellis()
        for strategy in (AntCycle(), AntDensity(), AntQuantity()):
            with self.subTest(strategy=type(strategy).__name__):
                solver = AntColonySolver(deposit=strategy, rng=random.Random(1))
                self.assertEqual(len(solver.solve(trellis).path), trellis.length)

    def test_pheromone_alone_still_finds_the_optimum(self):
        """The regression test for the reinforcement loop.

        With ``beta=0`` the cost heuristic is switched off, so the only thing
        that can steer the ants is what previous generations deposited. If the
        pheromone update were inert this would be a blind random walk.
        """
        trellis = sample_trellis()
        hits = sum(
            self.solver(seed=seed, beta=0.0, evaporation=0.2).solve(trellis).path
            == OPTIMAL_PATH
            for seed in range(20)
        )
        self.assertGreaterEqual(hits, 18)

    def test_pheromone_concentrates_on_the_optimal_edges(self):
        trellis = sample_trellis()
        table = PheromoneTable(trellis)
        rng = random.Random(0)
        deposit = AntCycle()

        for _ in range(50):
            walks = [Ant(trellis, table, 1.0, 0.0, rng).walk() for _ in range(10)]
            table.evaporate(0.2)
            for walk in walks:
                deposit.deposit(table, trellis, walk)

        best = table.level(0, 0, OPTIMAL_PATH[0])
        rivals = [
            table.level(0, 0, tag)
            for tag in range(trellis.tag_count)
            if tag != OPTIMAL_PATH[0]
        ]
        self.assertGreater(best, 100 * max(rivals))

    def test_scales_past_the_three_by_three_case(self):
        """Guards the old rank/tag confusion: words and tags must be independent."""
        rng = random.Random(2024)
        trellis = random_trellis(rng, tag_count=5, length=8)
        self.assertEqual(len(TAGS), 3)  # the sample model is square; this one is not

        solution = self.solver(seed=1, generations=80).solve(trellis)
        self.assertEqual(len(solution.path), 8)
        self.assertTrue(all(0 <= tag < 5 for tag in solution.path))


if __name__ == "__main__":
    unittest.main()
