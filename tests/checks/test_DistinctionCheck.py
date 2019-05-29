from unittest import TestCase

from pandas import DataFrame

from shift_detector.checks.DistinctionCheck import DistinctionCheck
from shift_detector.precalculations.Store import Store


class TestCreateDetector(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.precalculation = DistinctionCheck()

    def test_run(self):
        report = self.precalculation.run(self.store)

        with self.subTest("Test shifted columns"):
            self.assertCountEqual(report.shifted_columns, ['shift'])

        with self.subTest("Test examined columns"):
            self.assertCountEqual(report.examined_columns, ['shift', 'no_shift'])

        with self.subTest("Test additional information"):
            self.assertEqual(report.information['F1 score df1'], 1.0)
            self.assertEqual(report.information['F1 score df2'], 1.0)
