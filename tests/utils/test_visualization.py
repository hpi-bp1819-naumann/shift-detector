import unittest
from unittest import mock

import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from mock import MagicMock
from pandas.util.testing import assert_frame_equal

from shift_detector.utils.visualization import plot_cumulative_step_ratio_histogram, plot_binned_ratio_histogram, \
    plot_categorical_horizontal_ratio_histogram, calculate_value_ratios


class TestVisualizationUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.num_column1 = pd.Series(list(range(10)))
        self.num_column2 = pd.Series(list(range(9, 19)))
        self.cat_column1 = pd.Series(['blue'] * 5 + ['red'] * 4 + ['green'] * 1, name='color')
        self.cat_column2 = pd.Series(['blue'] * 3 + ['red'] * 7 + ['green'] * 0, name='color')

    def test_cumulative_hist(self):
        mock_axes = MagicMock()
        cumsum1, cumsum2 = plot_cumulative_step_ratio_histogram(mock_axes, columns=(self.num_column1, self.num_column2),
                                                                bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_axes.plot.called)
        np.testing.assert_almost_equal(desired=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], actual=list(cumsum1))
        np.testing.assert_almost_equal(desired=[0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0], actual=list(cumsum2))

    def test_ratio_hist(self):
        mock_axes = MagicMock()
        plot_binned_ratio_histogram(mock_axes, columns=(self.num_column1, self.num_column2),
                                    bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_axes.bar.called)

    def test_value_ratios(self):
        result = calculate_value_ratios(columns=(self.cat_column1, self.cat_column2), top_k=2)
        solution = pd.DataFrame([[0.5, 0.3], [0.4, 0.7]], index=['blue', 'red'], columns=['color 1', 'color 2'])
        assert_frame_equal(solution, result)

    @mock.patch('shift_detector.utils.visualization.pd.DataFrame.plot')
    def test_categorical_horizontal_ratio_hist(self, mock_plot):
        mock_axes = MagicMock()
        plot_categorical_horizontal_ratio_histogram(mock_axes, columns=(self.cat_column1, self.cat_column2))
        mock_plot.assert_called_with(kind='barh', fontsize='medium', ax=mock_axes)
