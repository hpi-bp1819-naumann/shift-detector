import math
import pprint as pp
import unittest

import pandas as pd

from Morpheus.precalculations.simple_precalculation import SimplePrecalculation
from Morpheus.precalculations.store import Store


class TestSimplePrecalculation(unittest.TestCase):

    def setUp(self) -> None:
        self.precalculation = SimplePrecalculation()

        numerical_df_1 = pd.DataFrame.from_dict(
            {'col_1': range(100), 'col_2': list(range(50)) * 2, 'col_3': range(0, 200, 2)})
        numerical_df_2 = pd.DataFrame.from_dict(
            {'col_1': range(1, 101), 'col_2': [8] + [None] * 99, 'col_3': list(range(50, 100)) * 2})
        categorical_df_1 = pd.DataFrame(['red', 'blue', 'blue', 'green', 'green', 'green'] * 20)
        categorical_df_2 = pd.DataFrame(['red', 'green', 'green', 'green', 'green', 'green'] * 20)

        self.store_numerical = Store(numerical_df_1, numerical_df_2)
        self.store_categorical = Store(categorical_df_1, categorical_df_2)

    def test_minmax_metrics(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['value_range']['df1'], 99.0)
        self.assertEqual(comparison_numeric['col_1']['value_range']['df2'], 99.0)
        self.assertEqual(comparison_numeric['col_2']['value_range']['df1'], 49.0)
        self.assertEqual(comparison_numeric['col_2']['value_range']['df2'], 0.0)
        self.assertEqual(comparison_numeric['col_3']['value_range']['df1'], 198.0)
        self.assertEqual(comparison_numeric['col_3']['value_range']['df2'], 49.0)

    def test_quartile_metrics(self):
        numerical_df_1 = pd.DataFrame(list(range(12)), columns=['col_1'])
        numerical_df_2 = pd.DataFrame(list(range(1, 20, 2)), columns=['col_1'])
        new_numerical_store = Store(numerical_df_1, numerical_df_2)

        comparison_numeric = self.precalculation.process(new_numerical_store)['numerical_comparison']
        self.assertAlmostEqual(comparison_numeric['col_1']['quartile_1']['df1'], numerical_df_1['col_1'].quantile(.25))
        self.assertEqual(comparison_numeric['col_1']['quartile_3']['df1'], numerical_df_1['col_1'].quantile(.75))
        self.assertEqual(comparison_numeric['col_1']['median']['df1'], numerical_df_1['col_1'].quantile(.5))
        self.assertAlmostEqual(comparison_numeric['col_1']['quartile_1']['df2'], numerical_df_2['col_1'].quantile(.25))
        self.assertEqual(comparison_numeric['col_1']['quartile_3']['df2'], numerical_df_2['col_1'].quantile(.75))
        self.assertEqual(comparison_numeric['col_1']['median']['df2'], numerical_df_2['col_1'].quantile(.5))

    def test_mean_std(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['mean']['df1'], 49.5)
        self.assertEqual(comparison_numeric['col_2']['mean']['df1'], 24.5)
        self.assertEqual(comparison_numeric['col_3']['mean']['df1'], 99.0)
        self.assertEqual(comparison_numeric['col_1']['mean']['df2'], 50.5)
        self.assertEqual(comparison_numeric['col_2']['mean']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['mean']['df2'], 74.5)

        self.assertAlmostEqual(comparison_numeric['col_1']['std']['df1'], 29.011, places=3)
        self.assertAlmostEqual(comparison_numeric['col_2']['std']['df1'], 14.5035, places=3)
        self.assertAlmostEqual(comparison_numeric['col_3']['std']['df1'], 58.0229, places=3)
        self.assertAlmostEqual(comparison_numeric['col_1']['std']['df2'], 29.011, places=3)
        self.assertTrue(math.isnan(comparison_numeric['col_2']['std']['df2']))
        self.assertAlmostEqual(comparison_numeric['col_3']['std']['df2'], 14.5035, places=3)

    def test_uniqueness(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df1'], 0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df2'], 0)

    def test_num_distinct(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['num_distinct']['df1'], 100.0)
        self.assertEqual(comparison_numeric['col_2']['num_distinct']['df1'], 50.0)
        self.assertEqual(comparison_numeric['col_3']['num_distinct']['df1'], 100.0)
        self.assertEqual(comparison_numeric['col_1']['num_distinct']['df2'], 100.0)
        self.assertEqual(comparison_numeric['col_2']['num_distinct']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['num_distinct']['df2'], 50.0)

    def test_completeness(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_1']['completeness']['df2'], 1.00)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df2'], 0.01)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df2'], 1.00)

    def test_categorical_values(self):
        comparison_categorical = self.precalculation.process(self.store_categorical)['categorical_comparison']
        pp.pprint(comparison_categorical)

        self.assertEqual(comparison_categorical[0]['blue']['df1'], 1/3)
        self.assertEqual(comparison_categorical[0]['green']['df1'], 0.5)
        self.assertEqual(comparison_categorical[0]['green']['df2'], 5/6)
        self.assertEqual(comparison_categorical[0]['red']['df1'], 1/6)
        self.assertEqual(comparison_categorical[0]['red']['df2'], 1/6)
