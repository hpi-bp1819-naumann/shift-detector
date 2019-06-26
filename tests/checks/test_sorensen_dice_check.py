import unittest

import pandas as pd
import mock

from shift_detector.checks.sorensen_dice_check import SorensenDiceCheck, SorensenDiceReport
from shift_detector.precalculations.n_gram import NGramType
from shift_detector.precalculations.store import Store


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

    def test_calls_embedding_distance_report(self):
        self.assertEqual(self.report.__class__, SorensenDiceReport)

    def test_threshold(self):
        report2 = SorensenDiceCheck(ngram_type=NGramType.character, n=3, threshold=1.1).run(self.store)
        self.assertEqual(report2.shifted_columns, [])


class TestEmbeddingDistanceReport(unittest.TestCase):

    def setUp(self):
        result = {'Col1': (0.1, 0.2, 1.0),
                  'Col2': (0.2, 0.2, 0.2)}
        self.report = SorensenDiceReport('Test Check', ['Col1', 'Col2'], ['Col1'], information=(0.1, result))

    @mock.patch('shift_detector.checks.sorensen_dice_check.display')
    def test_report_calls_display(self, mock_display):
        self.report.print_information()
        self.assertTrue(mock_display.called)
