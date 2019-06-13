from unittest import TestCase

import numpy as np
from pandas import DataFrame

from shift_detector.precalculations.binning_precalculation import BinningPrecalculation
from shift_detector.precalculations.store import Store


class TestBinningPrecalculation(TestCase):

    def setUp(self):
        self.n = 5
        self.df1 = DataFrame({'num': np.arange(1, 101), 'cat': [0] * 100})
        self.df2 = DataFrame({'num': np.arange(101, 201), 'cat': [0] * 100})

        self.store = Store(self.df1, self.df2)
        self.precalculation = BinningPrecalculation(5)

    def test_eq(self):
        with self.subTest("Test inequality"):
            self.assertNotEqual("Not a binning", self.precalculation)

        with self.subTest("Test equality"):
            other_precalculation = BinningPrecalculation(5)
            self.assertEqual(self.precalculation, other_precalculation)

    def test_hash(self):
        other_precalculation = BinningPrecalculation(5)
        self.assertEqual(hash(self.precalculation), hash(other_precalculation))

    def test_process(self):
        df1_binned, df2_binned = self.precalculation.process(self.store)

        with self.subTest("Test numerical binning"):
            column_name = 'num_binned'
            df1_binned_expected_values = [key for key, value in df1_binned[column_name].value_counts().items()
                                          if value > 0]
            df2_binned_expected_values = [key for key, value in df2_binned[column_name].value_counts().items()
                                          if value > 0]

            self.assertEqual(3, len(df1_binned_expected_values))
            self.assertEqual(3, len(df2_binned_expected_values))

        with self.subTest("Non binning on categorical, numerical data"):
            column_name = 'cat'
            df1_binned_expected_values = [key for key, value in df1_binned[column_name].value_counts().items()
                                          if value > 0]
            df2_binned_expected_values = [key for key, value in df2_binned[column_name].value_counts().items()
                                          if value > 0]

            self.assertEqual(1, len(df1_binned_expected_values))
            self.assertEqual(1, len(df2_binned_expected_values))
