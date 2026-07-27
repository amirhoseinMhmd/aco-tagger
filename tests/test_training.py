"""Corpus reading, model estimation, evaluation, and the training CLI."""

from __future__ import annotations

import io
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import mkdtemp

from pos_aco.adapters.corpora import CorpusError, read_tagged_file
from pos_aco.domain.corpus import split
from pos_aco.application.evaluation import Accuracy, evaluate
from pos_aco.adapters.csv_model import CsvLanguageModelLoader
from pos_aco.domain.model import UNKNOWN_WORD
from pos_aco.application.tagger import PosTagger
from pos_aco.train import main as train_main
from pos_aco.application.training import ModelEstimator
from pos_aco.search.viterbi import ViterbiSolver

from .support import TREEBANK_DIR

CORPUS = [
    [("The", "ART"), ("dog", "NOUN"), ("runs", "VERB")],
    [("The", "ART"), ("cat", "NOUN"), ("runs", "VERB")],
]


class TempDirectoryTest(unittest.TestCase):
    def temp_dir(self) -> Path:
        directory = Path(mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        return directory


class ReadTaggedFileTest(TempDirectoryTest):
    def write(self, text: str) -> Path:
        path = self.temp_dir() / "corpus.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_one_sentence_per_line(self):
        sentences = read_tagged_file(self.write("the/ART dog/NOUN\nit/PRON runs/VERB\n"))
        self.assertEqual(sentences[0], [("the", "ART"), ("dog", "NOUN")])
        self.assertEqual(len(sentences), 2)

    def test_splits_on_the_last_separator_so_slashes_survive(self):
        sentences = read_tagged_file(self.write("and/or/CC\n"))
        self.assertEqual(sentences[0], [("and/or", "CC")])

    def test_skips_blank_lines(self):
        self.assertEqual(len(read_tagged_file(self.write("a/X\n\n\nb/Y\n"))), 2)

    def test_rejects_a_token_without_a_tag(self):
        with self.assertRaises(CorpusError) as caught:
            read_tagged_file(self.write("the dog/NOUN\n"))
        self.assertIn("word/TAG", str(caught.exception))

    def test_rejects_a_missing_file(self):
        with self.assertRaises(CorpusError):
            read_tagged_file(self.temp_dir() / "absent.txt")

    def test_rejects_an_empty_file(self):
        with self.assertRaises(CorpusError):
            read_tagged_file(self.write("\n\n"))


class SplitTest(unittest.TestCase):
    def test_holds_out_the_tail(self):
        training, held_out = split(list(range(10)), holdout=0.2)
        self.assertEqual(training, list(range(8)))
        self.assertEqual(held_out, [8, 9])

    def test_a_zero_holdout_keeps_everything_for_training(self):
        training, held_out = split(list(range(10)), holdout=0.0)
        self.assertEqual(len(training), 10)
        self.assertEqual(held_out, [])

    def test_rejects_a_holdout_of_one_or_more(self):
        with self.assertRaises(ValueError):
            split(list(range(10)), holdout=1.0)


class ModelEstimatorTest(unittest.TestCase):
    def estimate(self, **settings):
        defaults = {"smoothing": 0.0, "unknown_threshold": 0}
        return ModelEstimator(**{**defaults, **settings}).estimate(CORPUS)

    def test_orders_tags_predictably(self):
        self.assertEqual(self.estimate().tags, ("ART", "NOUN", "VERB"))

    def test_emissions_are_conditioned_on_the_tag(self):
        model = self.estimate()
        # 'the' is the only ART, so P(the|ART) = 2/2; 'dog' is 1 of 2 NOUNs.
        self.assertAlmostEqual(model.emission("the")[0], 1.0)
        self.assertAlmostEqual(model.emission("dog")[1], 0.5)
        self.assertAlmostEqual(model.emission("dog")[0], 0.0)

    def test_emission_rows_are_sparse(self):
        row = self.estimate().emission("runs")
        self.assertEqual(row, (0.0, 0.0, 1.0))

    def test_lowercases_through_the_shared_normalizer(self):
        """Training and tagging must agree, or lookups silently miss."""
        model = self.estimate()
        self.assertIn("the", model.emissions)
        self.assertNotIn("The", model.emissions)

    def test_unsmoothed_initial_is_the_raw_first_tag_distribution(self):
        self.assertEqual(self.estimate().initial, (1.0, 0.0, 0.0))

    def test_smoothing_reserves_mass_for_unseen_tags(self):
        model = self.estimate(smoothing=1.0)
        # ART starts both sentences: (2 + 1) / (2 + 1*3)
        self.assertAlmostEqual(model.initial[0], 0.6)
        self.assertAlmostEqual(model.initial[1], 0.2)

    def test_transitions_follow_observed_pairs(self):
        model = self.estimate()
        self.assertAlmostEqual(model.transition(0)[1], 1.0)  # ART -> NOUN
        self.assertAlmostEqual(model.transition(1)[2], 1.0)  # NOUN -> VERB

    def test_a_tag_that_never_precedes_anything_falls_back_to_uniform(self):
        # VERB is always sentence-final here, so it has no outgoing counts.
        model = self.estimate()
        self.assertEqual(model.transition(2), (1 / 3, 1 / 3, 1 / 3))

    def test_rare_words_are_folded_into_the_unknown_entry(self):
        model = self.estimate(unknown_threshold=1)
        self.assertTrue(model.handles_unknown_words)
        self.assertNotIn("dog", model.emissions)  # seen once
        self.assertIn("the", model.emissions)  # seen twice
        self.assertEqual(model.emission("neverseen"), model.emission(UNKNOWN_WORD))

    def test_drops_bare_punctuation_tokens(self):
        model = ModelEstimator(smoothing=0.0, unknown_threshold=0).estimate(
            [[("The", "ART"), ("dog", "NOUN"), (".", ".")]]
        )
        self.assertNotIn(".", model.tags)
        self.assertEqual(len(model.emissions), 2)

    def test_rejects_an_empty_corpus(self):
        with self.assertRaises(ValueError):
            ModelEstimator().estimate([])

    def test_rejects_negative_settings(self):
        with self.assertRaises(ValueError):
            ModelEstimator(smoothing=-1.0)
        with self.assertRaises(ValueError):
            ModelEstimator(unknown_threshold=-1)


class AccuracyTest(unittest.TestCase):
    def test_splits_the_score_by_whether_the_word_was_known(self):
        score = Accuracy(
            correct=8, total=10, correct_known=7, total_known=8, seconds=0.0
        )
        self.assertAlmostEqual(score.overall, 0.8)
        self.assertAlmostEqual(score.known, 0.875)
        self.assertEqual(score.total_unknown, 2)
        self.assertAlmostEqual(score.unknown, 0.5)

    def test_reports_zero_rather_than_dividing_by_zero(self):
        score = Accuracy(0, 0, 0, 0, 0.0)
        self.assertEqual(score.overall, 0.0)
        self.assertEqual(score.unknown, 0.0)


class EvaluateTest(unittest.TestCase):
    def test_recovers_the_training_data_it_was_estimated_from(self):
        model = ModelEstimator(smoothing=0.0, unknown_threshold=0).estimate(CORPUS)
        tagger = PosTagger(model, ViterbiSolver())

        score = evaluate(tagger, model, CORPUS)
        self.assertEqual(score.total, 6)
        self.assertEqual(score.overall, 1.0)
        self.assertEqual(score.total_unknown, 0)

    def test_counts_out_of_vocabulary_words_as_unknown(self):
        model = ModelEstimator(smoothing=1.0, unknown_threshold=1).estimate(CORPUS)
        tagger = PosTagger(model, ViterbiSolver())

        score = evaluate(tagger, model, [[("The", "ART"), ("bird", "NOUN")]])
        self.assertEqual(score.total, 2)
        self.assertEqual(score.total_known, 1)  # 'the' survived; 'bird' never seen

    def test_limit_caps_the_number_of_sentences_scored(self):
        model = ModelEstimator(smoothing=0.0, unknown_threshold=0).estimate(CORPUS)
        tagger = PosTagger(model, ViterbiSolver())
        self.assertEqual(evaluate(tagger, model, CORPUS, limit=1).total, 3)


class TrainCommandLineTest(TempDirectoryTest):
    def run_train(self, *arguments) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = train_main(list(arguments))
        return code, out.getvalue(), err.getvalue()

    def corpus_file(self) -> Path:
        path = self.temp_dir() / "corpus.txt"
        path.write_text(
            "the/ART dog/NOUN runs/VERB\n" * 5 + "the/ART cat/NOUN sits/VERB\n" * 5,
            encoding="utf-8",
        )
        return path

    def test_writes_a_model_that_loads_back(self):
        output = self.temp_dir() / "model"
        code, out, _ = self.run_train(
            "--corpus", str(self.corpus_file()),
            "--out", str(output),
            "--holdout", "0",
            "--evaluate", "none",
        )
        self.assertEqual(code, 0)
        self.assertIn("written to", out)

        model = CsvLanguageModelLoader(output).load()
        self.assertEqual(model.tags, ("ART", "NOUN", "VERB"))

    def test_reports_held_out_accuracy(self):
        code, out, _ = self.run_train(
            "--corpus", str(self.corpus_file()),
            "--out", str(self.temp_dir() / "model"),
            "--holdout", "0.2",
            "--evaluate", "viterbi",
        )
        self.assertEqual(code, 0)
        self.assertIn("viterbi", out)
        self.assertIn("overall", out)

    def test_reports_a_missing_corpus_without_a_traceback(self):
        code, _, err = self.run_train(
            "--corpus", str(self.temp_dir() / "absent.txt"),
            "--out", str(self.temp_dir() / "model"),
        )
        self.assertEqual(code, 2)
        self.assertIn("absent.txt", err)

    def test_an_extreme_holdout_still_leaves_one_training_sentence(self):
        """split() floors the holdout, so training is never emptied."""
        code, out, _ = self.run_train(
            "--corpus", str(self.corpus_file()),
            "--out", str(self.temp_dir() / "model"),
            "--holdout", "0.999",
            "--evaluate", "none",
        )
        self.assertEqual(code, 0)
        self.assertIn("1 training / 9 held out", out)


class BundledTreebankModelTest(unittest.TestCase):
    """The committed model in data/treebank, which the CLI uses by default."""

    @classmethod
    def setUpClass(cls):
        if not (TREEBANK_DIR / "lexicon.csv").is_file():
            raise unittest.SkipTest("data/treebank has not been generated")
        cls.model = CsvLanguageModelLoader(TREEBANK_DIR).load()

    def test_covers_a_realistic_tagset(self):
        self.assertGreater(len(self.model.tags), 30)

    def test_models_unknown_words(self):
        self.assertTrue(self.model.handles_unknown_words)

    def test_tags_a_sentence_of_words_it_never_saw(self):
        tagger = PosTagger(self.model, ViterbiSolver())
        tagging = tagger.tag("the quixotic zzzblargh fizzled")
        self.assertEqual(len(tagging.tags), 4)

    def test_tags_a_plain_sentence_sensibly(self):
        tagger = PosTagger(self.model, ViterbiSolver())
        tags = dict(tagger.tag("the company said it will sell the unit").pairs)
        self.assertEqual(tags["the"], "DT")
        self.assertEqual(tags["company"], "NN")
        self.assertEqual(tags["said"], "VBD")
        self.assertEqual(tags["sell"], "VB")


if __name__ == "__main__":
    unittest.main()
