import unittest
from shift_detector.checks.EmbeddingDistanceCheck import EmbeddingDistanceCheck
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestEmbeddingDistanceCheck(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef'],
                                 'col2': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy',
                                          'z1', '23', '45', '67', '89', '10', '11', '12', '13', '14']
                                 })
        self.df2 = pd.DataFrame({'col1': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy',
                                          'z1', '23', '45', '67', '89', '10', '11', '12', '13', '14'],
                                 'col2': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy',
                                          'z1', '23', '45', '67', '89', '10', '11', '12', '13', '15']})

        self.store = Store(self.df1, self.df2)
        self.report = EmbeddingDistanceCheck(model='word2vec').run(self.store)

    def test_examined_columns(self):
        self.assertIn('col1', self.report.examined_columns)
        self.assertIn('col2', self.report.examined_columns)

    def test_shifted_columns(self):
        self.assertIn('col1', self.report.shifted_columns)
        self.assertNotIn('col2', self.report.shifted_columns)

    def test_explanation_existence(self):
        self.assertNotEqual(self.report.explanation, '')
