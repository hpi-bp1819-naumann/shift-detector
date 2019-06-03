import unittest
from shift_detector.precalculations.LdaEmbedding import LdaEmbedding
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestLdaEmbedding(unittest.TestCase):

    def setUp(self):
        self.lda1 = LdaEmbedding(n_topics=2, n_iter=1, random_state=0)
        self.lda2 = LdaEmbedding(n_topics=2, n_iter=1, random_state=0)

        # https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/amazon-fine-food-reviews.zip/2
        reviews = pd.read_csv('../../data/Reviews.csv')

        self.df1 = pd.DataFrame(reviews[reviews['Score'] == 5][:50].reset_index()['Text'])
        self.df2 = pd.DataFrame(reviews[reviews['Score'] == 1][:50].reset_index()['Text'])
        self.store = Store(self.df1, self.df2)

    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: LdaEmbedding(n_topics=0))

    def test_eq(self):
        self.assertTrue(self.lda1 == self.lda2)

    def test_hash(self):
        self.assertEqual(hash(self.lda1), hash(self.lda2))

    def test_process(self):
        res1, res2 = self.lda1.process(self.store)
        self.assertTrue(res1['topic'], [1,1,0,0,1,1,1,1,1,1,
                                        1,1,0,0,0,0,1,1,0,1,
                                        1,0,1,0,0,1,1,1,1,1,
                                        1,1,1,1,1,1,1,0,1,1,
                                        0,1,1,1,1,1,0,1,1,1])
