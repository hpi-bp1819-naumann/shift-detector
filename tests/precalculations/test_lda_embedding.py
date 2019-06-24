import unittest
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.precalculations.store import Store
import pandas as pd
import os


class TestLdaEmbedding(unittest.TestCase):

    def setUp(self):
        self.lda1 = LdaEmbedding(cols=['text'], n_topics=2, n_iter=1, random_state=2, lib='sklearn')
        self.lda2 = LdaEmbedding(cols=['text'], n_topics=2, n_iter=1, random_state=2, lib='sklearn')
        self.lda3 = LdaEmbedding(cols=['text'], n_topics=2, n_iter=1, random_state=2, lib='gensim')
        self.lda4 = LdaEmbedding(cols=['text', 'abc'], n_topics=2, n_iter=1, random_state=2, lib='sklearn')
        print(os.path.dirname(os.path.abspath(__file__)))
        print(os.getcwd())
        self.poems = pd.read_csv("../data/poems.csv")
        self.phrases = pd.read_csv("../data/phrases.csv")

        self.df1 = pd.DataFrame(self.poems, columns=['text'])
        self.df2 = pd.DataFrame(self.phrases, columns=['text'])
        self.store = Store(self.df1, self.df2)

    def test_exception_for_n_topics(self):
        self.assertRaises(ValueError, lambda: LdaEmbedding(cols='', n_topics=0))
        self.assertRaises(TypeError, lambda: LdaEmbedding(cols='', n_topics=1.5))

    def test_exception_for_lib(self):
        self.assertRaises(ValueError, lambda: LdaEmbedding(cols='', lib='?'))

    def test_eq(self):
        self.assertEqual(self.lda1, self.lda2)
        self.assertNotEqual(self.lda3, self.lda4)

    def test_hash(self):
        self.assertEqual(hash(self.lda1), hash(self.lda2))
        self.assertNotEqual(hash(self.lda3), hash(self.lda4))

    def test_process(self):
        res1, res2, topic_words_all_cols, all_models, all_dtms, all_vecs = self.lda1.process(self.store)
        res3, res4, topic_words_all_cols, all_models, all_corpora, all_dicts = self.lda3.process(self.store)

        self.assertTrue(res1['topics text'].equals(pd.Series([0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                              0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                                                              0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])))

        self.assertTrue(res3['topics text'].equals(pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0,
                                                              1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
                                                              0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
                                                              0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1])))

    def test_column_exception_in_process(self):
        self.assertRaises(ValueError, lambda: self.lda4.process(self.store))
