import os
import unittest
from unittest.mock import MagicMock

import pandas as pd

from shift_detector.Detector import Detector
from shift_detector.checks.DummyCheck import DummyCheck, DummyReport


class TestCreateDetector(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ['Jones LLC'] * 100}
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1
        self.path = 'test.csv'
        self.df1.to_csv(self.path)

    def test_init(self):
        with self.subTest("Test successful initialization with csv paths"):
            Detector(self.path, self.path)

        with self.subTest("Test unsuccessful initialization with wrong df1 parameter"):
            no_df = 0
            self.assertRaises(Exception, Detector, no_df, no_df)

        with self.subTest("Test unsuccessful initialization with wrong df2 parameter"):
            no_df = 0
            self.assertRaises(Exception, Detector, self.df1, no_df)

    def tearDown(self):
        os.remove(self.path)


class TestDetector(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ['Jones LLC']}
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1

        self.detector = Detector(df1=self.df1, df2=self.df2)

    def test_run(self):
        with self.subTest("Test unsuccessful run"):
            self.assertRaises(Exception, self.detector.run)

        with self.subTest("Test successful run"):
            self.detector.checks_to_run = [DummyCheck(), DummyCheck()]
            self.detector.run()
            self.assertEquals(len(self.detector.check_reports), 2)

    def test_add_check(self):
        with self.subTest("Test with single check"):
            check = DummyCheck()
            self.detector.add_checks(check)
            self.assertEquals(len(self.detector.checks_to_run), 1)

        with self.subTest("Test with not a check"):
            no_check = "No check"
            self.assertRaises(Exception, self.detector.add_checks, no_check)

    def test_add_checks(self):
        with self.subTest("Test with list of checks"):
            checks = [DummyCheck(), DummyCheck()]
            self.detector.add_checks(checks)
            self.assertEquals(len(self.detector.checks_to_run), 2)

    def test_evaluate(self):
        mock = MagicMock()
        mock.print_report = MagicMock()
        self.detector.check_reports = [mock]
        self.detector.evaluate()
        mock.print_report.assert_called_with()
