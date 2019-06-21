import unittest
from unittest import mock

from mock import MagicMock

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck


class TestSimpleStatisticalCheck(unittest.TestCase):

    @mock.patch('shift_detector.checks.statistical_checks.statistical_check.plt')
    def test_all_figures_are_called_and_plot_is_shown(self, mock_plt):
        figures = [MagicMock(), MagicMock(), MagicMock()]
        SimpleStatisticalCheck.plot_all_columns(figures)
        for figure in figures:
            self.assertTrue(figure.called)
        mock_plt.show.assert_called_with()