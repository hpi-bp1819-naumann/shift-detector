from unittest import TestCase

from pandas import DataFrame

from shift_detector.precalculations.DistinctionPrecalculation import DistinctionPrecalculation
from shift_detector.precalculations.Store import Store


class TestCreateDetector(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.precalculation = DistinctionPrecalculation(['shift', 'no_shift'])

    def test_process(self):
        calculation = self.precalculation.process(self.store)
        with self.subTest("Test prediction"):
            y_true = calculation['y_true']
            y_pred = calculation['y_pred']
            self.assertTrue(y_pred.equals(y_true))

        with self.subTest("Test relevant columns"):
            shifted_columns = calculation['relevant_columns']
            self.assertCountEqual(shifted_columns, ['shift'])

    def test_equal(self):
        with self.subTest("Test equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift', 'shift'])
            self.assertEqual(self.precalculation, other_precalculation)

        with self.subTest("Test equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift'])
            self.assertNotEqual(self.precalculation, other_precalculation)

    def test_hash(self):
        with self.subTest("Test hash equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift', 'shift'])
            self.assertEqual(self.precalculation.__hash__(), other_precalculation.__hash__())

        with self.subTest("Test hash equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift'])
            self.assertNotEqual(self.precalculation.__hash__(), other_precalculation.__hash__())
