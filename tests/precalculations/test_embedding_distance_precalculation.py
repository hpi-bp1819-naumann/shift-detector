import unittest
from morpheus.precalculations.embedding_distance_precalculation import EmbeddingDistancePrecalculation
from morpheus.precalculations.store import Store
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class TestEmbeddingDistancePrecalculation(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy',
                                          'z1', '23', '45', '67', '89', '10', '11', '12', '13', '14']})

        self.store = Store(self.df1, self.df2)

        w2v = Word2Vec(size=50, window=5, min_count=1, workers=4)
        self.te1 = EmbeddingDistancePrecalculation(model='word2vec')
        self.te2 = EmbeddingDistancePrecalculation(model='word2vec')
        self.te3 = EmbeddingDistancePrecalculation(model='fasttext')
        self.te4 = EmbeddingDistancePrecalculation(trained_model=w2v)
        self.te5 = EmbeddingDistancePrecalculation(trained_model=w2v)

    def test_result(self):
        result = self.te1.process(self.store)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result['col1']), 3)
        self.assertEqual(result['col1'][0], 0.0)
        self.assertGreater(result['col1'][1], 0.0)
        self.assertGreater(result['col1'][2], 0.0)

    def test_eq(self):
        self.assertEqual(self.te1, self.te2)
        self.assertEqual(self.te4, self.te5)
        self.assertNotEqual(self.te2, self.te3)
        self.assertNotEqual(self.te2, self.te4)

    def test_hash(self):
        self.assertEqual(hash(self.te1), hash(self.te2))
        self.assertEqual(hash(self.te4), hash(self.te5))

    def test_join_and_normalize_vectors(self):
        ser1 = pd.Series([[7, 8, 9], [2, 3, 4], [0, 0, 0], [3, 5, 2]])
        self.assertTrue(np.array_equal(self.te1.sum_and_normalize_vectors(ser1), np.array([3.0, 4.0, 3.75])))

    def test_error_on_small_dataframe(self):
        df3 = pd.DataFrame({'col1': ['ab', 'hi', 'jk', 'lm', 'no', 'pq', 'rs', 'tu', 'vw', 'xy', '12', '34']})
        store2 = Store(self.df1, df3)
        self.assertRaises(ValueError, lambda: self.te2.process(store2))
