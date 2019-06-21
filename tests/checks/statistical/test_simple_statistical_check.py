import unittest
from unittest import mock

from matplotlib.figure import Figure
from mock import MagicMock

from shift_detector.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from shift_detector.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck


class TestSimpleStatisticalCheck(unittest.TestCase):

    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.plt.figure')
    def test_all_plot_functions_are_called_and_plot_is_shown(self, mock_plt_figure):
        mock_figure = MagicMock(autospec=Figure)
        mock_plt_figure.return_value = mock_figure
        plot_functions = [MagicMock(), MagicMock(), MagicMock()]
        for check, height in [(CategoricalStatisticalCheck(), 7.2), (NumericalStatisticalCheck(), 10.8)]:
            with self.subTest(check=check):
                check.plot_all_columns(plot_functions)
                mock_plt_figure.assert_called_with(figsize=(10, height), tight_layout=True)
                for func in plot_functions:
                    self.assertTrue(func.called)
                mock_figure.show.assert_called_with()

    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.plt.figure')
    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.gridspec')
    def test_grid_is_created(self, mock_grid, mock_plt_figure):
        plot_functions = [MagicMock(), MagicMock(), MagicMock()]
        for check, shape in [(CategoricalStatisticalCheck(), (2, 2)), (NumericalStatisticalCheck(), (3, 1))]:
            with self.subTest(check=check):
                check.plot_all_columns(plot_functions)
                mock_grid.GridSpec.assert_called_with(shape[0], shape[1])

    def test_correct_number_of_plot_functions(self):
        for sig_cols in [['col1', 'col2'], ['col1'], []]:
            with self.subTest(sig_cols=sig_cols):
                result = CategoricalStatisticalCheck().plot_functions(sig_cols, MagicMock(), MagicMock())
        self.assertEqual(len(sig_cols), len(result))
