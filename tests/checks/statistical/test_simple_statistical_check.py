import unittest
from unittest import mock

import pandas as pd
from matplotlib.figure import Figure
from mock import MagicMock

from shift_detector.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from shift_detector.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck
from shift_detector.checks.statistical_checks.text_metadata_statistical_check import TextMetadataStatisticalCheck
from shift_detector.precalculations.store import Store
from shift_detector.utils.visualization import PlotData


class TestSimpleStatisticalCheck(unittest.TestCase):

    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.plt')
    def test_all_plot_functions_are_called_and_plot_is_shown(self, mock_plt):
        plot_data = [PlotData(MagicMock(), 1), PlotData(MagicMock(), 2), PlotData(MagicMock(), 3)]
        for check, height in [(CategoricalStatisticalCheck(), 30.0), (NumericalStatisticalCheck(), 30.0)]:
            with self.subTest(check=check):
                check.plot_all_columns(plot_data)
                mock_plt.figure.assert_called_with(figsize=(12, height), tight_layout=True)
                for func, rows in plot_data:
                    self.assertTrue(func.called)
                mock_plt.show.assert_called_with()

    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.plt.figure')
    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.gridspec')
    def test_grid_is_created(self, mock_grid, mock_plt_figure):
        plot_data = [PlotData(MagicMock(), 1), PlotData(MagicMock(), 2), PlotData(MagicMock(), 3)]
        for check, shape in [(CategoricalStatisticalCheck(), (6, 1)), (NumericalStatisticalCheck(), (6, 1))]:
            with self.subTest(check=check):
                check.plot_all_columns(plot_data)
                mock_grid.GridSpec.assert_called_with(shape[0], shape[1])

    def test_correct_number_of_plot_functions(self):
        for sig_cols in [['col1', 'col2'], ['col1'], []]:
            with self.subTest(sig_cols=sig_cols):
                result = CategoricalStatisticalCheck().plot_data(sig_cols, MagicMock(), MagicMock())
        self.assertEqual(len(sig_cols), len(result))

    def test_size_adjustment(self):
        df1 = pd.DataFrame([0] * 20)
        df2 = pd.DataFrame([0] * 12)
        for check, solution in [(CategoricalStatisticalCheck(sample_size=10), (10, 10)),
                                (CategoricalStatisticalCheck(sample_size=15), (15, 12)),
                                (NumericalStatisticalCheck(use_equal_dataset_sizes=True), (12, 12)),
                                (TextMetadataStatisticalCheck(sample_size=15, use_equal_dataset_sizes=True), (12, 12))]:
            with self.subTest(solution=solution):
                result1, result2 = check.adjust_dataset_sizes(df1, df2)
                self.assertEqual(len(result1), solution[0])
                self.assertEqual(len(result2), solution[1])

    def test_column_order_in_report(self):
        df1 = pd.DataFrame([[1, 0]] * 10, columns=['abc', 'def'])
        df2 = pd.DataFrame([[0, 1]] * 10, columns=['abc', 'def'])
        store = Store(df1, df2)
        for check in [CategoricalStatisticalCheck(), NumericalStatisticalCheck()]:
            with self.subTest(check=check):
                result = check.run(store)
                self.assertEqual('abc', result.examined_columns[0])
                self.assertEqual('abc', result.shifted_columns[0])
                self.assertEqual(result.examined_columns, result.shifted_columns)
