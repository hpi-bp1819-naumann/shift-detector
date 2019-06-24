import unittest
from shift_detector.checks.lda_check import LdaCheck
from shift_detector.precalculations.store import Store
import pandas as pd
import logging


class TestLdaCheck(unittest.TestCase):

    def setUp(self):
        self.lda_report1 = LdaCheck(significance=0.1, n_topics=2, n_iter=1, random_state=2)
        self.lda_report2 = LdaCheck(significance=0.1, n_topics=2, n_iter=1, random_state=2)
        self.lda_report3 = LdaCheck(cols=['text'], significance=0.1, n_topics=2, n_iter=1, random_state=2)
        self.lda_report4 = LdaCheck(cols=['abcd'], significance=0.1, n_topics=2, n_iter=1, random_state=2)

        logging.error(os.path.dirname(os.path.abspath(__file__)))
        logging.error(os.getcwd())
        self.poems = pd.read_csv("../data/poems.csv")
        self.phrases = pd.read_csv("../data/phrases.csv")

        self.df1 = pd.DataFrame(self.poems, columns=['text'])
        self.df2 = pd.DataFrame(self.phrases, columns=['text'])
        self.store = Store(self.df1, self.df2)

    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: LdaCheck(significance=0.0))

    def test_exception_on_non_float(self):
        self.assertRaises(TypeError, lambda: LdaCheck(significance=42))
        self.assertRaises(TypeError, lambda: LdaCheck(significance='abcd'))

    def test_run(self):
        with self.subTest("Test successful run without specifying the 'cols' parameter"):
            report = self.lda_report1.run(self.store)
            self.assertAlmostEqual(report.explanation['Topic 1 diff in column text'], 0.405)
            self.assertAlmostEqual(report.explanation['Topic 2 diff in column text'], -0.405)

        with self.subTest("Test successful run with specifying the 'cols' parameter"):
            report = self.lda_report3.run(self.store)

            self.assertAlmostEqual(report.explanation['Topic 1 diff in column text'], 0.405)
            self.assertAlmostEqual(report.explanation['Topic 2 diff in column text'], -0.405)

        with self.subTest("Test unsuccessful run with specifying a wrong 'cols' parameter"):
            self.assertRaises(ValueError, lambda: self.lda_report4.run(self.store))
