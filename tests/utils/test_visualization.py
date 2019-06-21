import unittest

import pandas as pd
import numpy as np
from mock import MagicMock

from shift_detector.utils.visualization import plot_cumulative_step_ratio_histogram, plot_binned_ratio_histogram


class TestVisualizationUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.column1 = pd.Series(list(range(10)))
        self.column2 = pd.Series(list(range(9, 19)))

    def test_cumulative_hist(self):
        mock_axes = MagicMock()
        cumsum1, cumsum2 = plot_cumulative_step_ratio_histogram(mock_axes, columns=(self.column1, self.column2),
                                                                bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_axes.plot.called)
        np.testing.assert_almost_equal(desired=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], actual=list(cumsum1))
        np.testing.assert_almost_equal(desired=[0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0], actual=list(cumsum2))

    def test_ratio_hist(self):
        mock_axes = MagicMock()
        plot_binned_ratio_histogram(mock_axes, columns=(self.column1, self.column2),
                                    bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_axes.bar.called)
