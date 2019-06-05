import unittest
from shift_detector.precalculations.SorensenDicePrecalculation import SorensenDicePrecalculations
from shift_detector.precalculations.NGram import NGramType
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestSorensenDicePrecalculation(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab ', 'hij', 'jkl', 'lmn', 'nop', 'pqr', 'rst', 'tuv',
                                          'vwx', 'xyz', 'z12', '234', '456', '678', '890', 'zyx',
                                          'xwv', 'vut', 'tsr', 'rqp']})

        self.store = Store(self.df1, self.df2)
        self.result = SorensenDicePrecalculations(ngram_type=NGramType.character, n=3).process(self.store)

    def test_result(self):
        self.assertEqual(len(self.result), 1)
        self.assertEqual(self.result['col1'], (1.0, 0.0, (2 / 20) / 7))   # ngrams get normalized during join

    def test_eq(self):
        sd1 = SorensenDicePrecalculations(ngram_type=NGramType.character, n=3)
        sd2 = SorensenDicePrecalculations(ngram_type=NGramType.character, n=3)
        sd3 = SorensenDicePrecalculations(ngram_type=NGramType.character, n=2)
        sd4 = SorensenDicePrecalculations(ngram_type=NGramType.word, n=3)
        self.assertEqual(sd1, sd2)
        self.assertNotEqual(sd1, sd3)
        self.assertNotEqual(sd1, sd4)

    def test_hash(self):
        sd1 = SorensenDicePrecalculations(ngram_type=NGramType.character, n=3)
        sd2 = SorensenDicePrecalculations(ngram_type=NGramType.character, n=3)
        self.assertEqual(hash(sd1), hash(sd2))

    def test_count_fragments(self):
        ngram1 = {'abc': 6, 'bcd':4, 'def': 3}
        self.assertEqual(SorensenDicePrecalculations.count_fragments(ngram1), 13)

    def test_calculate_sdc(self):
        ngram1 = {'abc': 6, 'bcd':4, 'def': 3}
        ngram2 = {'abc': 2, 'def': 3, 'efg': 4}
        self.assertEqual(SorensenDicePrecalculations.calculate_sdc(ngram1, ngram2), 10 / 22)

    def test_join_and_normalze_ngrams(self):
        ngram_ser = pd.Series([{'abc': 6, 'bcd':4, 'def': 3}, {'abc': 2, 'def': 3, 'efg': 4}])
        result = SorensenDicePrecalculations.join_and_normalize_ngrams(ngram_ser)
        self.assertDictEqual(result, {'abc': 4, 'bcd': 2, 'def': 3, 'efg': 2})
        self.assertDictEqual(SorensenDicePrecalculations.join_and_normalize_ngrams(pd.Series([])), {})

    def test_error_on_small_dataframe(self):
        df3 = pd.DataFrame({'col1': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy', '12', '34']})
        store2 = Store(self.df1, df3)
        self.assertRaises(ValueError, lambda: SorensenDicePrecalculations(ngram_type=NGramType.character, n=3)
                          .process(store2))
