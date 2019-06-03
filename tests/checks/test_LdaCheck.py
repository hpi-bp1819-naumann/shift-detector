import unittest
from shift_detector.precalculations.LdaEmbedding import LdaEmbedding
from shift_detector.checks.LdaCheck import LdaCheck
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestLdaCheck(unittest.TestCase):

    def setUp(self):
        self.lda_report1 = LdaCheck(significance=10)
        self.lda_report2 = LdaCheck(significance=10)

        self.lda = LdaEmbedding(n_topics=2, n_iter=1, random_state=0)

        reviews = pd.read_csv('../../data/Reviews.csv')

        self.df1 = pd.DataFrame(reviews[reviews['Score'] == 5][:500].reset_index()['Text'])
        self.df2 = pd.DataFrame(reviews[reviews['Score'] == 1][:500].reset_index()['Text'])
        self.store = Store(self.df1, self.df2)

    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: LdaCheck(significance=0))

    def test_run(self):
        report = self.lda_report1.run(self.store)

        self.assertTrue(report.explanation['Topic 0 diff'] == 31.6)
        self.assertTrue(report.explanation['Topic 1 diff'] == -31.6)

