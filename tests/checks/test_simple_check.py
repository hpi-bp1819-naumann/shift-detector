import unittest
import pandas as pd
import pprint as pp

from shift_detector.Utils import ColumnType
from shift_detector.checks.SimpleCheck import SimpleCheck, SimpleCheckReport


class TestSimpleCheck(unittest.TestCase):

    def setUp(self) -> None:
        self.check = SimpleCheck()

        numerical_df_1 = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['col_1', 'col_2', 'col_3'])
        numerical_df_2 = pd.DataFrame([[7, 8, 9], [10, 11, 12]], columns=['col_1', 'col_2', 'col_3'])
        categorical_df_1 = pd.DataFrame(['red', 'blue'])
        categorical_df_2 = pd.DataFrame(['blue', 'orange'])

        self.check.data[ColumnType.numerical] = [numerical_df_1, numerical_df_2]
        self.check.data[ColumnType.categorical] = [categorical_df_1, categorical_df_2]

    def test_basic_functions(self):
        self.assertEqual(self.check.name(), 'SimpleCheck')
        self.assertEqual(self.check.report_class(), SimpleCheckReport)

    def test_run_with_empty_dataframes(self):
        self.check.data[ColumnType.numerical] = [pd.DataFrame([]), pd.DataFrame([])]
        self.check.data[ColumnType.categorical] = [pd.DataFrame([]), pd.DataFrame([])]
        self.assertEqual({'categorical_comparison': {}, 'numerical_comparison': {}}, self.check.run())

    def test_quantile_metrics(self):
        comparison_numeric = self.check.run()['numerical_comparison']
        pp.pprint(comparison_numeric)
        self.assertEqual(comparison_numeric['col_1']['min']['df1'], 1)
        self.assertEqual(comparison_numeric['col_1']['min']['df2'], 4)


