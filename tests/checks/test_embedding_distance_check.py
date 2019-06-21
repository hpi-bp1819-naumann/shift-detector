import unittest
import mock
from morpheus.checks.embedding_distance_check import EmbeddingDistanceCheck, EmbeddingDistanceReport
from morpheus.precalculations.store import Store
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

    def test_calls_embedding_distance_report(self):
        self.assertEqual(self.report.__class__, EmbeddingDistanceReport)


class TestEmbeddingDistanceReport(unittest.TestCase):

    def setUp(self):
        result = {'Col1': (0.1, 0.2, 20.0),
                  'Col2': (0.2, 0.2, 0.2)}
        self.report = EmbeddingDistanceReport('Test Check', ['Col1', 'Col2'], ['Col1'], information=result)

    @mock.patch('morpheus.checks.embedding_distance_check.display')
    def test_report_calls_display(self, mock_display):
        self.report.print_information()
        self.assertTrue(mock_display.called)
