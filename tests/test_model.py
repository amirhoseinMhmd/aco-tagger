"""Model invariants and CSV loading."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from pos_aco.adapters.csv_model import CsvLanguageModelLoader, CsvLanguageModelWriter
from pos_aco.domain.model import UNKNOWN_WORD, ModelError, UnknownWordError

from .support import DATA_DIR, EMISSIONS, INITIAL, TAGS, TRANSITIONS, sample_model


class LanguageModelTest(unittest.TestCase):
    def test_exposes_tag_set(self):
        model = sample_model()
        self.assertEqual(model.tags, TAGS)
        self.assertEqual(model.tag_count, 3)

    def test_looks_up_emissions_and_transitions(self):
        model = sample_model()
        self.assertEqual(model.emission("john"), (0.8, 0.1, 0.1))
        self.assertEqual(model.transition(2), (0.5, 0.4, 0.1))

    def test_translates_a_path_to_tag_names(self):
        self.assertEqual(sample_model().names_of((0, 2, 1)), ("NOUN", "ART", "VERB"))

    def test_rejects_unknown_words_with_a_clear_message(self):
        with self.assertRaises(UnknownWordError) as caught:
            sample_model().emission("missing")
        self.assertIn("missing", str(caught.exception))

    def test_a_model_without_an_unknown_entry_says_so(self):
        self.assertFalse(sample_model().handles_unknown_words)

    def test_falls_back_to_the_unknown_entry_when_present(self):
        model = sample_model(
            emissions={**EMISSIONS, UNKNOWN_WORD: (0.2, 0.7, 0.1)}
        )
        self.assertTrue(model.handles_unknown_words)
        self.assertEqual(model.emission("neverseen"), (0.2, 0.7, 0.1))

    def test_known_words_still_win_over_the_unknown_entry(self):
        model = sample_model(
            emissions={**EMISSIONS, UNKNOWN_WORD: (0.2, 0.7, 0.1)}
        )
        self.assertEqual(model.emission("john"), (0.8, 0.1, 0.1))

    def test_rejects_a_transition_matrix_of_the_wrong_height(self):
        with self.assertRaises(ModelError):
            sample_model(transitions=TRANSITIONS[:2])

    def test_rejects_a_row_of_the_wrong_width(self):
        with self.assertRaises(ModelError):
            sample_model(initial=(0.5, 0.5))

    def test_rejects_values_outside_the_unit_interval(self):
        with self.assertRaises(ModelError):
            sample_model(emissions={"john": (1.4, 0.1, 0.1)})

    def test_rejects_duplicate_tags(self):
        with self.assertRaises(ModelError):
            sample_model(tags=("NOUN", "NOUN", "ART"))

    def test_rejects_an_empty_lexicon(self):
        with self.assertRaises(ModelError):
            sample_model(emissions={})


class CsvLoadingTest(unittest.TestCase):
    def test_loads_the_bundled_sample_model(self):
        model = CsvLanguageModelLoader(DATA_DIR).load()
        self.assertEqual(model.tags, TAGS)
        self.assertEqual(model.emissions, EMISSIONS)
        self.assertEqual(model.transitions, TRANSITIONS)
        self.assertEqual(model.initial, INITIAL)

    def test_omitted_lexicon_entries_read_back_as_zero(self):
        """The sparse format stores only non-zero cells."""
        directory = self._write_model(lexicon="word,tag,probability\njohn,NOUN,0.8\n")
        model = CsvLanguageModelLoader(directory).load()
        self.assertEqual(model.emission("john"), (0.8, 0.0, 0.0))

    def test_round_trips_through_the_writer(self):
        directory = Path(mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        original = sample_model()

        CsvLanguageModelWriter(directory).save(original)
        self.assertEqual(CsvLanguageModelLoader(directory).load(), original)

    def test_reports_a_missing_file_by_path(self):
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(DATA_DIR / "nowhere").load()
        self.assertIn("initial.csv", str(caught.exception))

    def test_rejects_tag_columns_that_disagree_between_files(self):
        directory = self._write_model(transitions="NOUN,VERB,ADJ\n0.6,0.2,0.2\n")
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(directory).load()
        self.assertIn("do not match", str(caught.exception))

    def test_rejects_a_lexicon_with_the_wrong_header(self):
        directory = self._write_model(lexicon="word,NOUN,VERB,ART\njohn,0.8,0.1,0.1\n")
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(directory).load()
        self.assertIn("expected columns", str(caught.exception))

    def test_rejects_a_lexicon_naming_an_unknown_tag(self):
        directory = self._write_model(lexicon="word,tag,probability\njohn,ADJ,0.8\n")
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(directory).load()
        self.assertIn("unknown tag", str(caught.exception))

    def test_rejects_a_ragged_row(self):
        directory = self._write_model(lexicon="word,tag,probability\njohn,NOUN\n")
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(directory).load()
        self.assertIn("expected 3", str(caught.exception))

    def test_rejects_a_non_numeric_probability(self):
        directory = self._write_model(lexicon="word,tag,probability\njohn,NOUN,x\n")
        with self.assertRaises(ModelError):
            CsvLanguageModelLoader(directory).load()

    def test_rejects_an_empty_lexicon_file(self):
        directory = self._write_model(lexicon="word,tag,probability\n")
        with self.assertRaises(ModelError) as caught:
            CsvLanguageModelLoader(directory).load()
        self.assertIn("no lexicon entries", str(caught.exception))

    def _write_model(self, **overrides) -> Path:
        """A throwaway model directory, with one file swapped out."""
        files = {
            "lexicon": "word,tag,probability\njohn,NOUN,0.8\njohn,VERB,0.1\n",
            "transitions": "NOUN,VERB,ART\n0.6,0.2,0.2\n0.2,0.1,0.7\n0.5,0.4,0.1\n",
            "initial": "NOUN,VERB,ART\n0.6,0.1,0.3\n",
        }
        files.update(overrides)

        directory = Path(mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        for name, content in files.items():
            (directory / f"{name}.csv").write_text(content, encoding="utf-8")
        return directory


if __name__ == "__main__":
    unittest.main()
