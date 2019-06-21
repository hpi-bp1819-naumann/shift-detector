from unittest import TestCase

from pandas import DataFrame

from Morpheus.checks.distinction_check import DistinctionCheck
from Morpheus.precalculations.store import Store


class TestDistinctionCheck(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'small_shift': ['A', 'B'] * 50, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'small_shift': ['B', 'C'] * 50, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.check = DistinctionCheck()

    def test_init(self):
        with self.subTest("Test wrong columns"):
            self.assertRaises(TypeError, lambda: DistinctionCheck(['shift', 0]))

        with self.subTest("Test wrong num epochs"):
            self.assertRaises(TypeError, lambda: DistinctionCheck(num_epochs='wrong'))
            self.assertRaises(ValueError, lambda: DistinctionCheck(num_epochs=0))

        with self.subTest("Test wrong relative threshold"):
            self.assertRaises(TypeError, lambda: DistinctionCheck(relative_threshold='wrong'))
            self.assertRaises(ValueError, lambda: DistinctionCheck(relative_threshold=-1))

    def test_run(self):
        report = self.check.run(self.store)

        with self.subTest("Test shifted columns"):
            self.assertCountEqual(report.shifted_columns, ['shift', 'small_shift'])

        with self.subTest("Test examined columns"):
            self.assertCountEqual(report.examined_columns, ['shift', 'small_shift', 'no_shift'])
