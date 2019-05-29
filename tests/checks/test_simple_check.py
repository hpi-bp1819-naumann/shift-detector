import math
import pprint as pp
import unittest

import pandas as pd

from shift_detector.precalculations.SimplePrecalculation import SimplePrecalculation
from shift_detector.precalculations.Store import Store


class TestSimplePrecalculation(unittest.TestCase):

    def setUp(self) -> None:
        self.precalculation = SimplePrecalculation()

        numerical_df_1 = pd.DataFrame([[1, 2, 3], [4, 2, 6]], columns=['col_1', 'col_2', 'col_3'])
        numerical_df_2 = pd.DataFrame([[7, 8, 8], [10, None, 8]], columns=['col_1', 'col_2', 'col_3'])
        categorical_df_1 = pd.DataFrame(['red', 'blue', 'blue', 'green', 'green', 'green'])
        categorical_df_2 = pd.DataFrame(['red', 'green', 'green', 'green', 'green', 'green'])

        self.store_numerical = Store(numerical_df_1, numerical_df_2)
        self.store_categorical = Store(categorical_df_1, categorical_df_2)

    def test_run_with_empty_dataframes(self):
        empty_store = Store(pd.DataFrame([]), pd.DataFrame([]))
        self.assertEqual({'categorical_comparison': {}, 'numerical_comparison': {}}, self.precalculation.process(empty_store))

    def test_minmax_metrics(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['min']['df1'], 1)
        self.assertEqual(comparison_numeric['col_2']['min']['df1'], 2)
        self.assertEqual(comparison_numeric['col_3']['min']['df1'], 3)
        self.assertEqual(comparison_numeric['col_1']['min']['df2'], 7)
        self.assertEqual(comparison_numeric['col_2']['min']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['min']['df2'], 8)

        self.assertEqual(comparison_numeric['col_1']['max']['df1'], 4)
        self.assertEqual(comparison_numeric['col_2']['max']['df1'], 2)
        self.assertEqual(comparison_numeric['col_3']['max']['df1'], 6)
        self.assertEqual(comparison_numeric['col_1']['max']['df2'], 10)
        self.assertEqual(comparison_numeric['col_2']['max']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['max']['df2'], 8)

    def test_quartile_metrics(self):
        numerical_df_1 = pd.DataFrame([[0], [1], [2], [3], [4]], columns=['col_1'])
        numerical_df_2 = pd.DataFrame([[1], [3]], columns=['col_1'])
        new_numerical_store = Store(numerical_df_1, numerical_df_2)

        comparison_numeric = self.precalculation.process(new_numerical_store)['numerical_comparison']
        self.assertEqual(comparison_numeric['col_1']['quartile_1']['df1'], 1)
        self.assertEqual(comparison_numeric['col_1']['quartile_3']['df1'], 3)
        self.assertEqual(comparison_numeric['col_1']['median']['df1'], 2)

    def test_mean_std(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['mean']['df1'], 2.5)
        self.assertEqual(comparison_numeric['col_2']['mean']['df1'], 2)
        self.assertEqual(comparison_numeric['col_3']['mean']['df1'], 4.5)
        self.assertEqual(comparison_numeric['col_1']['mean']['df2'], 8.5)
        self.assertEqual(comparison_numeric['col_2']['mean']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['mean']['df2'], 8)

        self.assertEqual(round(comparison_numeric['col_1']['std']['df1'], 3), 2.121)
        self.assertEqual(round(comparison_numeric['col_2']['std']['df1'], 3), 0)
        self.assertEqual(round(comparison_numeric['col_3']['std']['df1'], 3), 2.121)
        self.assertEqual(round(comparison_numeric['col_1']['std']['df2'], 3), 2.121)
        self.assertTrue(math.isnan(comparison_numeric['col_2']['std']['df2']))
        self.assertEqual(round(comparison_numeric['col_3']['std']['df2'], 3), 0)

    def test_uniqueness(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df1'], 0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df2'], 0)

    def test_distinctness(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['num_distinct']['df1'], 2.0)
        self.assertEqual(comparison_numeric['col_2']['num_distinct']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['num_distinct']['df1'], 2.0)
        self.assertEqual(comparison_numeric['col_1']['num_distinct']['df2'], 2.0)
        self.assertEqual(comparison_numeric['col_2']['num_distinct']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['num_distinct']['df2'], 1.0)

    def test_completeness(self):
        comparison_numeric = self.precalculation.process(self.store_numerical)['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['completeness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_1']['completeness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df2'], 0.5)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df2'], 1.0)

    def test_categorical_values(self):
        comparison_categorical = self.precalculation.process(self.store_categorical)['categorical_comparison']
        pp.pprint(comparison_categorical)

        self.assertEqual(comparison_categorical[0]['blue']['df1'], 1/3)
        self.assertEqual(comparison_categorical[0]['green']['df1'], 0.5)
        self.assertEqual(comparison_categorical[0]['green']['df2'], 5/6)
        self.assertEqual(comparison_categorical[0]['red']['df1'], 1/6)
        self.assertEqual(comparison_categorical[0]['red']['df2'], 1/6)
