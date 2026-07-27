"""Using the tagger for languages other than English.

The model, trellis and solvers treat words as opaque strings, so a new language
is a retraining job. What is genuinely language-specific lives in
:mod:`pos_aco.tokenizer`, and that is most of what is tested here.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from pos_aco.adapters.corpora import CorpusError, read_conllu
from pos_aco.adapters.csv_model import CsvLanguageModelLoader, CsvLanguageModelWriter
from pos_aco.application.tagger import PosTagger
from pos_aco.domain.tokenizer import (
    DefaultTokenizer,
    PersianTokenizer,
    TurkishTokenizer,
    get_tokenizer,
    strip_punctuation,
)
from pos_aco.application.training import ModelEstimator
from pos_aco.search.viterbi import ViterbiSolver

# "today the weather is snowy" / "he read the book", with the kaf and yeh
# deliberately spelled inconsistently, the way real Persian text is.
PERSIAN_CORPUS = [
    [("امروز", "ADV"), ("هوا", "N"), ("برفی", "ADJ"), ("است", "V")],
    [("امروز", "ADV"), ("هوا", "N"), ("سرد", "ADJ"), ("است", "V")],
    [("او", "PRO"), ("كتاب", "N"), ("را", "POST"), ("خواند", "V")],
    [("او", "PRO"), ("کتاب", "N"), ("را", "POST"), ("دید", "V")],
]

CONLLU = """\
# sent_id = 1
# text = Vamos al mar
1\tVamos\tir\tVERB\tVMIP1P0\t_\t0\troot\t_\t_
2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_
2\ta\ta\tADP\tSPS00\t_\t4\tcase\t_\t_
3\tel\tel\tDET\tDA0MS0\t_\t4\tdet\t_\t_
4\tmar\tmar\tNOUN\tNCMS000\t_\t1\tobl\t_\t_
4.1\t_\t_\t_\t_\t_\t_\t_\t_\t_

# sent_id = 2
1\tEl\tel\tDET\tDA0MS0\t_\t2\tdet\t_\t_
2\tmar\tmar\tNOUN\tNCMS000\t_\t0\troot\t_\t_
"""


class TempDirectoryTest(unittest.TestCase):
    def temp_dir(self) -> Path:
        directory = Path(mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        return directory


class StripPunctuationTest(unittest.TestCase):
    """Punctuation is a Unicode category, not an ASCII list."""

    def test_strips_non_latin_punctuation(self):
        self.assertEqual(strip_punctuation("است،"), "است")  # Arabic comma
        self.assertEqual(strip_punctuation("بله؟"), "بله")  # Arabic question mark
        self.assertEqual(strip_punctuation("المدرسة؛"), "المدرسة")  # Arabic semicolon
        self.assertEqual(strip_punctuation("है।"), "है")  # Devanagari danda
        self.assertEqual(strip_punctuation("です。"), "です")  # ideographic full stop

    def test_strips_dashes_and_quotation_marks(self):
        self.assertEqual(strip_punctuation("--"), "")
        self.assertEqual(strip_punctuation("«"), "")
        self.assertEqual(strip_punctuation('"quoted"'), "quoted")

    def test_leaves_the_inside_of_a_word_alone(self):
        self.assertEqual(strip_punctuation("dit-il"), "dit-il")
        self.assertEqual(strip_punctuation("don't"), "don't")


class DefaultTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DefaultTokenizer()

    def test_handles_whitespace_delimited_scripts(self):
        self.assertEqual(
            self.tokenizer.tokenize("امروز هوا برفی است، بله؟"),
            ("امروز", "هوا", "برفی", "است", "بله"),
        )
        self.assertEqual(
            self.tokenizer.tokenize("यह एक वाक्य है।"), ("यह", "एक", "वाक्य", "है")
        )

    def test_cannot_segment_scripts_without_spaces(self):
        """A known limit, recorded so it is not mistaken for a bug.

        Chinese, Japanese, Thai and friends need a segmenter in front of this.
        """
        self.assertEqual(len(self.tokenizer.tokenize("我喜欢自然语言处理。")), 1)
        self.assertEqual(len(self.tokenizer.tokenize("ฉันชอบภาษาไทย")), 1)

    def test_case_folds_for_matching(self):
        self.assertEqual(self.tokenizer.normalize("Ünïcode"), "ünïcode")


class PersianTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PersianTokenizer()

    def test_unifies_arabic_and_persian_kaf(self):
        # U+0643 vs U+06A9 - the same word, two codepoints.
        self.assertEqual(
            self.tokenizer.normalize("كتاب"), self.tokenizer.normalize("کتاب")
        )

    def test_unifies_arabic_and_farsi_yeh(self):
        # U+064A vs U+06CC.
        self.assertEqual(
            self.tokenizer.normalize("مي‌رود"), self.tokenizer.normalize("می‌رود")
        )

    def test_unifies_arabic_indic_and_persian_digits(self):
        self.assertEqual(self.tokenizer.normalize("٥"), self.tokenizer.normalize("۵"))

    def test_keeps_the_zero_width_non_joiner_inside_a_word(self):
        """ZWNJ is part of the spelling, not punctuation."""
        self.assertIn("‌", self.tokenizer.normalize("می‌رود"))

    def test_removes_tatweel_which_is_pure_decoration(self):
        self.assertEqual(self.tokenizer.normalize("کـــتاب"), "کتاب")

    def test_the_default_tokenizer_does_not_unify_them(self):
        """Which is exactly why the Persian one exists."""
        default = DefaultTokenizer()
        self.assertNotEqual(default.normalize("كتاب"), default.normalize("کتاب"))


class TurkishTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TurkishTokenizer()

    def test_dotless_capital_i_folds_to_dotless_lowercase(self):
        self.assertEqual(self.tokenizer.normalize("ISTANBUL"), "ıstanbul")

    def test_dotted_capital_i_folds_to_plain_i(self):
        self.assertEqual(self.tokenizer.normalize("İzmir"), "izmir")

    def test_the_default_tokenizer_gets_this_wrong(self):
        self.assertEqual(DefaultTokenizer().normalize("ISTANBUL"), "istanbul")


class TokenizerRegistryTest(unittest.TestCase):
    def test_looks_tokenizers_up_by_name(self):
        self.assertIsInstance(get_tokenizer("persian"), PersianTokenizer)
        self.assertIsInstance(get_tokenizer("turkish"), TurkishTokenizer)
        self.assertIsInstance(get_tokenizer(), DefaultTokenizer)

    def test_rejects_an_unknown_name_and_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as caught:
            get_tokenizer("klingon")
        self.assertIn("persian", str(caught.exception))


class ConlluTest(TempDirectoryTest):
    def write(self, text: str) -> Path:
        path = self.temp_dir() / "corpus.conllu"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_sentences_split_by_blank_lines(self):
        sentences = read_conllu(self.write(CONLLU))
        self.assertEqual(len(sentences), 2)

    def test_uses_the_universal_tag_column_by_default(self):
        first = read_conllu(self.write(CONLLU))[0]
        self.assertEqual(first[0], ("Vamos", "VERB"))
        self.assertEqual([tag for _, tag in first], ["VERB", "ADP", "DET", "NOUN"])

    def test_can_use_the_treebank_specific_column_instead(self):
        first = read_conllu(self.write(CONLLU), tagset="xpos")[0]
        self.assertEqual(first[0], ("Vamos", "VMIP1P0"))

    def test_skips_multiword_ranges_that_would_double_count(self):
        words = [word for word, _ in read_conllu(self.write(CONLLU))[0]]
        self.assertNotIn("al", words)  # the 2-3 range covering "a" + "el"

    def test_skips_empty_nodes(self):
        self.assertEqual(len(read_conllu(self.write(CONLLU))[0]), 4)  # not 5

    def test_rejects_an_unknown_tagset(self):
        with self.assertRaises(CorpusError):
            read_conllu(self.write(CONLLU), tagset="lemma")

    def test_rejects_a_row_with_the_wrong_column_count(self):
        with self.assertRaises(CorpusError) as caught:
            read_conllu(self.write("1\tVamos\tir\tVERB\n"))
        self.assertIn("expected 10", str(caught.exception))

    def test_rejects_a_missing_file(self):
        with self.assertRaises(CorpusError):
            read_conllu(self.temp_dir() / "absent.conllu")


class TokenizerMetadataTest(TempDirectoryTest):
    """The tokenizer travels with the model, or train/serve skew comes back."""

    def train(self, tokenizer_name: str):
        tokenizer = get_tokenizer(tokenizer_name)
        model = ModelEstimator(
            smoothing=1.0, unknown_threshold=0, tokenizer=tokenizer
        ).estimate(PERSIAN_CORPUS)
        directory = self.temp_dir()
        CsvLanguageModelWriter(directory).save(model, tokenizer_name)
        return directory, model

    def test_folding_merges_the_two_spellings_into_one_entry(self):
        _, folded = self.train("persian")
        _, unfolded = self.train("default")
        self.assertEqual(len(folded.emissions) + 1, len(unfolded.emissions))

    def test_the_name_round_trips_through_metadata(self):
        directory, _ = self.train("persian")
        self.assertEqual(CsvLanguageModelLoader(directory).tokenizer_name(), "persian")
        self.assertEqual(
            CsvLanguageModelLoader(directory).metadata()["tokenizer"], "persian"
        )

    def test_a_model_without_metadata_falls_back_to_the_default(self):
        directory, _ = self.train("persian")
        (directory / "metadata.csv").unlink()
        self.assertEqual(CsvLanguageModelLoader(directory).tokenizer_name(), "default")

    def test_a_persian_model_tags_persian_text(self):
        directory, model = self.train("persian")
        loader = CsvLanguageModelLoader(directory)
        tagger = PosTagger(
            loader.load(),
            ViterbiSolver(),
            tokenizer=get_tokenizer(loader.tokenizer_name()),
        )
        tagging = tagger.tag("امروز هوا سرد است.")
        self.assertEqual(tagging.tags, ("ADV", "N", "ADJ", "V"))

    def test_either_spelling_of_a_word_reaches_the_same_entry(self):
        directory, _ = self.train("persian")
        loader = CsvLanguageModelLoader(directory)
        model, tokenizer = loader.load(), get_tokenizer(loader.tokenizer_name())
        self.assertEqual(
            model.emission(tokenizer.normalize("كتاب")),
            model.emission(tokenizer.normalize("کتاب")),
        )


if __name__ == "__main__":
    unittest.main()
