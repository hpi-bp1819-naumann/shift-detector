import unittest

import pandas as pd

from shift_detector.checks.SorensenDiceCheck import SorensenDiceCheck
from shift_detector.precalculations.NGram import NGramType
from shift_detector.precalculations.Store import Store


class TestSorensenDiceCheck(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab c', 'hij', 'jkl', 'lmn', 'nop', 'pqr', 'rst', 'tuv',
                                          'vwx', 'xyz', 'z12', '234', '456', '678', '890', 'zyx',
                                          'xwv', 'vut', 'tsr', 'rqp']})

        self.store = Store(self.df1, self.df2)
        self.report = SorensenDiceCheck(ngram_type=NGramType.character, n=3).run(self.store)

    def test_examined_columns(self):
        self.assertEqual(self.report.examined_columns, ['col1'])

    def test_shifted_columns(self):
        self.assertEqual(self.report.shifted_columns, ['col1'])

    def test_explanation_existence(self):
        self.assertNotEqual(self.report.explanation, '')

    def test_equal_datasets(self):
        store2 = Store(self.df2, self.df2)
        report2 = SorensenDiceCheck(ngram_type=NGramType.character, n=3).run(store2)
        self.assertEqual(report2.shifted_columns, [])
