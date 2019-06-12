import os
import unittest
from unittest.mock import MagicMock

import pandas as pd

from shift_detector.Detector import Detector
from shift_detector.checks.DummyCheck import DummyCheck


class TestCreateDetector(unittest.TestCase):

    def setUp(self):
        sales1 = {'brand': ['Jones LLC'] * 100}
        sales2 = {'brand': ['Merger LLC'] * 100}
        self.df1 = pd.DataFrame.from_dict(sales1)
        self.df2 = pd.DataFrame.from_dict(sales2)

        self.path1 = 'test1.csv'
        self.df1.to_csv(self.path1, index=False)
        self.path2 = 'test2.csv'
        self.df2.to_csv(self.path2, index=False)

    def test_init(self):
        with self.subTest("Test successful initialization with csv paths"):
            detector = Detector(self.path1, self.path2)
            self.assertTrue(self.df1.equals(detector.df1))
            self.assertTrue(self.df2.equals(detector.df2))

        with self.subTest("Test unsuccessful initialization with wrong df1 parameter"):
            no_df = 0
            self.assertRaises(Exception, Detector, no_df, no_df)

        with self.subTest("Test unsuccessful initialization with wrong df2 parameter"):
            no_df = 0
            self.assertRaises(Exception, Detector, self.df1, no_df)

    def tearDown(self):
        os.remove(self.path1)
        os.remove(self.path2)


class TestDetector(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ['Jones LLC'] * 10}
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1

        self.detector = Detector(df1=self.df1, df2=self.df2)

    def test_run(self):
        with self.subTest("Test unsuccessful run"):
            self.assertRaises(Exception, self.detector.run)

        with self.subTest("Test successful run"):
            self.detector.run(DummyCheck(), DummyCheck())
            self.assertEqual(len(self.detector.check_reports), 2)

    def test_evaluate(self):
        mock = MagicMock()
        mock.__str__ = MagicMock(return_value="")
        self.detector.check_reports = [mock]
        self.detector.evaluate()
        mock.__str__.assert_called_with()
