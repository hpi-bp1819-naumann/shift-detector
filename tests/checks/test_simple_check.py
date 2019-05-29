import math
import pprint as pp
import unittest

import pandas as pd

from shift_detector.checks.SimpleCheck import SimpleCheck
from shift_detector.preprocessors.Store import Store


class TestSimpleCheck(unittest.TestCase):

    def setUp(self) -> None:
        self.check = SimpleCheck()
        numerical_df_1 = pd.DataFrame.from_dict(
            {'col_1': range(100), 'col_2': list(range(50)) * 2, 'col_3': range(0, 200, 2)})
        numerical_df_2 = pd.DataFrame.from_dict(
            {'col_1': range(1, 101), 'col_2': [8] + [None] * 99, 'col_3': list(range(50, 100)) * 2})
        categorical_df_1 = pd.DataFrame(['red', 'blue', 'blue', 'green', 'green', 'green'] * 20)
        categorical_df_2 = pd.DataFrame(['red', 'green', 'green', 'green', 'green', 'green'] * 20)

        self.store_numerical = Store(numerical_df_1, numerical_df_2)
        self.store_categorical = Store(categorical_df_1, categorical_df_2)

    def test_run_with_empty_dataframes(self):
        empty_store = Store(pd.DataFrame([]), pd.DataFrame([]))
        self.assertEqual({'categorical_comparison': {}, 'numerical_comparison': {}}, self.check.run(empty_store).data)

    def test_minmax_metrics(self):
        comparison_numeric = self.check.run(self.store_numerical).data['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['min']['df1'], 0)
        self.assertEqual(comparison_numeric['col_2']['min']['df1'], 0)
        self.assertEqual(comparison_numeric['col_3']['min']['df1'], 0)
        self.assertEqual(comparison_numeric['col_1']['min']['df2'], 1)
        self.assertEqual(comparison_numeric['col_2']['min']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['min']['df2'], 50)

        self.assertEqual(comparison_numeric['col_1']['max']['df1'], 99)
        self.assertEqual(comparison_numeric['col_2']['max']['df1'], 49)
        self.assertEqual(comparison_numeric['col_3']['max']['df1'], 198)
        self.assertEqual(comparison_numeric['col_1']['max']['df2'], 100)
        self.assertEqual(comparison_numeric['col_2']['max']['df2'], 8)
        self.assertEqual(comparison_numeric['col_3']['max']['df2'], 99)

    def test_quartile_metrics(self):
        numerical_df_1 = pd.DataFrame([[0], [1], [2], [3], [4]], columns=['col_1'])
        numerical_df_2 = pd.DataFrame([[1], [3]], columns=['col_1'])
        new_numerical_store = Store(numerical_df_1, numerical_df_2)

        comparison_numeric = self.check.run(new_numerical_store).data['numerical_comparison']
        self.assertEqual(comparison_numeric['col_1']['quartile_1']['df1'], 1)
        self.assertEqual(comparison_numeric['col_1']['quartile_3']['df1'], 3)
        self.assertEqual(comparison_numeric['col_1']['median']['df1'], 2)

    def test_mean_std(self):
        comparison_numeric = self.check.run(self.store_numerical).data['numerical_comparison']

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
        comparison_numeric = self.check.run(self.store_numerical).data['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df1'], 0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_1']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['uniqueness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['uniqueness']['df2'], 0)

    def test_distinctness(self):
        comparison_numeric = self.check.run(self.store_numerical).data['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['distinctness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['distinctness']['df1'], 0.5)
        self.assertEqual(comparison_numeric['col_3']['distinctness']['df1'], 1.0)
        self.assertEqual(comparison_numeric['col_1']['distinctness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_2']['distinctness']['df2'], 1.0)
        self.assertEqual(comparison_numeric['col_3']['distinctness']['df2'], 0.5)

    def test_completeness(self):
        comparison_numeric = self.check.run(self.store_numerical).data['numerical_comparison']

        self.assertEqual(comparison_numeric['col_1']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df1'], 1.00)
        self.assertEqual(comparison_numeric['col_1']['completeness']['df2'], 1.00)
        self.assertEqual(comparison_numeric['col_2']['completeness']['df2'], 0.01)
        self.assertEqual(comparison_numeric['col_3']['completeness']['df2'], 1.00)

    def test_categorical_values(self):
        comparison_categorical = self.check.run(self.store_categorical).data['categorical_comparison']
        pp.pprint(comparison_categorical)

        self.assertEqual(comparison_categorical[0]['blue']['df1'], 1/3)
        self.assertEqual(comparison_categorical[0]['green']['df1'], 0.5)
        self.assertEqual(comparison_categorical[0]['green']['df2'], 5/6)
        self.assertEqual(comparison_categorical[0]['red']['df1'], 1/6)
        self.assertEqual(comparison_categorical[0]['red']['df2'], 1/6)
