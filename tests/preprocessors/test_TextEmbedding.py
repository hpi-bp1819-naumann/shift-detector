import unittest
from shift_detector.precalculations.TextEmbedding import TextEmbedding
from gensim.models import FastText
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestSorensenDice(unittest.TestCase):

    def setUp(self):
        df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        df2 = pd.DataFrame({'col1': ['ab ', 'hi ', 'jk ', 'lm ', 'no ', 'pq ', 'rs ', 'tu ',
                                          'vw ', 'xy ', 'z1 ', '23 ', '45 ', '67 ', '89 ']})
        self.store = Store(df1, df2)
        ft1 = FastText(size=50, window=5, min_count=1, workers=4)
        self.te1 = TextEmbedding(model='word2vec')
        self.te2 = TextEmbedding(model='word2vec')
        self.te3 = TextEmbedding(model='fasttext')
        self.te4 = TextEmbedding(trained_model=ft1)
        self.te5 = TextEmbedding(trained_model=ft1)

    def test_eq(self):
        self.assertEqual(self.te1, self.te2)
        self.assertEqual(self.te4, self.te5)
        self.assertNotEqual(self.te1, self.te3)
        self.assertNotEqual(self.te3, self.te4)

    def test_hash(self):
        self.assertEqual(hash(self.te1), hash(self.te2))
        self.assertEqual(hash(self.te4), hash(self.te5))

    def test_exception_in_init(self):
        self.assertRaises(ValueError, lambda: TextEmbedding())
        self.assertRaises(ValueError, lambda: TextEmbedding(model='fasttextt'))

    def test_result(self):
        df1, df2 = self.te1.process(self.store)
        for column in df1:
            for field in df1[column]:
                self.assertEqual(len(field), 300)
