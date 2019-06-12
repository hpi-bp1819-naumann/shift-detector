from unittest import TestCase

from pandas import DataFrame

from shift_detector.precalculations.distinction_precalculation import DistinctionPrecalculation
from shift_detector.precalculations.store import Store


class TestCreateDetector(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.precalculation = DistinctionPrecalculation(['shift', 'no_shift'], num_epochs=10)

    def test_init(self):
        with self.subTest("Test wrong columns"):
            self.assertRaises(TypeError, lambda: DistinctionPrecalculation(['shift', 0]))

        with self.subTest("Test wrong num epochs"):
            self.assertRaises(Exception, lambda: DistinctionPrecalculation(columns=['shift'], num_epochs='wrong'))
            self.assertRaises(Exception, lambda: DistinctionPrecalculation(columns=['shift'], num_epochs=0))

    def test_process(self):
        examined_column, calculation = self.precalculation.process(self.store)

        with self.subTest("Test examined column"):
            self.assertCountEqual(examined_column, ['shift', 'no_shift'])

        with self.subTest("Test prediction"):
            y_true = calculation['y_true']
            y_pred = calculation['y_pred']
            self.assertTrue(y_pred.equals(y_true))

        with self.subTest("Test base accuracy"):
            base_accuracy = calculation['base_accuracy']
            self.assertEqual(base_accuracy, 1.)

        with self.subTest("Test permuted accuracy"):
            shift_permuted_accuracy = calculation['permuted_accuracies']['shift']
            self.assertEqual(shift_permuted_accuracy, 0.)

            no_shift_permuted_accuracy = calculation['permuted_accuracies']['no_shift']
            self.assertEqual(no_shift_permuted_accuracy, 1.)

        with self.subTest("Test wrong columns"):
            prec_wrong_columns = DistinctionPrecalculation(['wrong_column'], num_epochs=10)
            self.assertRaises(Exception, lambda: prec_wrong_columns.process(self.store))

    def test_equal(self):
        with self.subTest("Test equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift', 'shift'])
            self.assertEqual(self.precalculation, other_precalculation)

        with self.subTest("Test inequality with other columns"):
            other_precalculation = DistinctionPrecalculation(['no_shift'], num_epochs=10)
            self.assertNotEqual(self.precalculation, other_precalculation)

        with self.subTest("Test inequality with another number of epochs"):
            other_precalculation = DistinctionPrecalculation(['no_shift', 'shift'], num_epochs=100)
            self.assertNotEqual(self.precalculation, other_precalculation)

        with self.subTest("Test inequality with another class"):
            other = "Not a DistinctPrecalculation"
            self.assertNotEqual(self.precalculation, other)

    def test_hash(self):
        with self.subTest("Test hash equality"):
            other_precalculation = DistinctionPrecalculation(['no_shift', 'shift'])
            self.assertEqual(self.precalculation.__hash__(), other_precalculation.__hash__())

        with self.subTest("Test hash inequality"):
            other_precalculation = DistinctionPrecalculation(['no_shift'])
            self.assertNotEqual(self.precalculation.__hash__(), other_precalculation.__hash__())
