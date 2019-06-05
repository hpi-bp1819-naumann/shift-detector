import unittest
from shift_detector.checks.EmbeddingDistanceCheck import EmbeddingDistanceCheck
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestEmbeddingDistanceCheck(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy',
                                          'z1', '23', '45', '67', '89', '10', '11', '12', '13', '14']})

        self.store = Store(self.df1, self.df2)
        self.report = EmbeddingDistanceCheck(model='word2vec').run(self.store)

    def test_examined_columns(self):
        self.assertEqual(self.report.examined_columns, {'col1'})

    def test_shifted_columns(self):
        self.assertEqual(self.report.shifted_columns, {'col1'})

    def test_explanation_existence(self):
        self.assertNotEqual(self.report.explanation, '')
