import os
import unittest
from unittest.mock import Mock, MagicMock, patch, call

import pandas as pd

from shift_detector.checks.check import Check
from shift_detector.detector import Detector


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
            detector = Detector(self.path1, self.path2, log_print=False)
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

        self.detector = Detector(df1=self.df1, df2=self.df2, log_print=False)

    def test_run(self):
        with self.subTest("Test unsuccessful run"):
            self.assertRaises(Exception, self.detector.run)

        with self.subTest("Test successful run"):
            check = Mock(spec=Check)
            check.run.return_value = 0
            self.detector.run(check, check)
            self.assertEqual(len(self.detector.check_reports), 2)
            self.assertEqual(check.run.call_count, 2)

        with self.subTest("Test run failing check"):
            mock = Mock(spec=Check)
            mock.run.side_effect = Mock(side_effect=Exception("Test Exception"))
            self.detector.run(mock)
            self.assertEqual(len(self.detector.check_reports), 1)
            error_msg = self.detector.check_reports[0].information['Exception']
            self.assertEqual(error_msg, "Test Exception")

    @patch('builtins.print')
    def test_evaluate(self, mocked_print):
        mock = MagicMock()
        mock.print_report = MagicMock()
        self.detector.check_reports = [mock]
        self.detector.evaluate()
        mock.print_report.assert_called_with()
