from unittest import TestCase

from pandas import DataFrame

from morpheus.precalculations.low_cardinality_precalculation import LowCardinalityPrecalculation
from morpheus.precalculations.store import Store


class TestLowCardinalityPrecalculation(TestCase):

    def setUp(self):
        self.precalculation = LowCardinalityPrecalculation()

    def test_equal(self):
        with self.subTest("Equality"):
            other_precalculation = LowCardinalityPrecalculation()
            self.assertEqual(self.precalculation, other_precalculation)

        with self.subTest("Inequality"):
            no_precalculation = 'no_precalculation'
            self.assertNotEqual(self.precalculation, no_precalculation)

    def test_hash(self):
        with self.subTest("Equality"):
            other_precalculation = LowCardinalityPrecalculation()
            self.assertEqual(self.precalculation.__hash__(), other_precalculation.__hash__())

        with self.subTest("Inequality"):
            no_precalculation = 'no_precalculation'
            self.assertNotEqual(self.precalculation.__hash__(), no_precalculation.__hash__())

    def test_process(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"] * 10,
                 'payment': [150.0, 200.0, 50.0, 10.0, 5.0, 150.0, 200.0, 50.0, 10.0, 5.0, 1.0] * 10,
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] * 10}
        df1 = DataFrame.from_dict(sales)
        df2 = df1

        store = Store(df1, df2)
        df1_processed, _, columns = self.precalculation.process(store)
        self.assertCountEqual(['brand', 'payment'], columns)
        self.assertTrue(df1_processed.equals(df1[['brand', 'payment']]))
