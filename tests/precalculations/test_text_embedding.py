import unittest
from shift_detector.precalculations.text_embedding_precalculation import TextEmbeddingPrecalculation
from gensim.models import FastText
from shift_detector.precalculations.store import Store
import pandas as pd


class TestTextEmbeddingPrecalculation(unittest.TestCase):

    def setUp(self):
        df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                     'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                     'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                     'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', '']})
        df2 = pd.DataFrame({'col1': ['ab ', 'hi ', 'jk ', 'lm ', 'no ', 'pq ', 'rs ', 'tu ',
                                     'vw ', 'xy ', 'z1 ', '23 ', '45 ', '67 ', '89 ']})
        self.store = Store(df1, df2)
        self.ft1 = FastText(size=50, window=5, min_count=1, workers=4)
        self.te1 = TextEmbeddingPrecalculation(model='word2vec', agg='sum')
        self.te2 = TextEmbeddingPrecalculation(model='word2vec', agg='sum')
        self.w2v_avg1 = TextEmbeddingPrecalculation(model='word2vec', agg='avg')
        self.w2v_avg2 = TextEmbeddingPrecalculation(model='word2vec', agg='avg')
        self.te3 = TextEmbeddingPrecalculation(model='fasttext', agg='sum')
        self.te4 = TextEmbeddingPrecalculation(trained_model=self.ft1, agg='sum')
        self.te5 = TextEmbeddingPrecalculation(trained_model=self.ft1, agg='sum')

        self.ft_no_agg = TextEmbeddingPrecalculation(trained_model=self.ft1, agg=None)

    def test_eq(self):
        self.assertEqual(self.te1, self.te2)
        self.assertEqual(self.te4, self.te5)
        self.assertEqual(self.w2v_avg1, self.w2v_avg2)
        self.assertNotEqual(self.te1, self.te3)
        self.assertNotEqual(self.te2, self.w2v_avg1)
        self.assertNotEqual(self.te3, self.te4)

    def test_hash(self):
        self.assertEqual(hash(self.te1), hash(self.te2))
        self.assertEqual(hash(self.w2v_avg1), hash(self.w2v_avg2))
        self.assertEqual(hash(self.te4), hash(self.te5))

    def test_exception_in_init(self):
        self.assertRaises(ValueError, lambda: TextEmbeddingPrecalculation())
        self.assertRaises(ValueError, lambda: TextEmbeddingPrecalculation(model='fasttextt'))

    def test_result(self):

        with self.subTest('Test sum'):
            df1, df2 = self.te1.process(self.store)
            for column in df1:
                for field in df1[column]:
                    self.assertEqual(100, len(field))

        with self.subTest('Test avg'):
            df1, df2 = self.w2v_avg1.process(self.store)
            for column in df1:
                for field in df1[column]:
                    self.assertEqual(100, len(field))

        with self.subTest('Test no agg'):
            df1, df2 = self.ft_no_agg.process(self.store)
            for column in df1:
                for field in df1[column]:
                    if not field.shape[0] >= 1:
                        print(field.shape)
                    self.assertTrue(field.shape[0] >= 1)
                    self.assertEqual(self.ft1.vector_size, field.shape[1])
