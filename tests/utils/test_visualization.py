import unittest
from unittest import mock

import pandas as pd
import numpy as np

from Morpheus.utils.visualization import plot_cumulative_step_ratio_histogram, plot_ratio_histogram


class TestVisualizationUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.column1 = pd.Series(list(range(10)))
        self.column2 = pd.Series(list(range(9, 19)))

    @mock.patch('Morpheus.utils.visualization.plt')
    def test_cumulative_hist(self, mock_plt):
        cumsum1, cumsum2 = plot_cumulative_step_ratio_histogram(self.column1, self.column2,
                                                                bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_plt.plot.called)
        np.testing.assert_almost_equal(desired=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], actual=list(cumsum1))
        np.testing.assert_almost_equal(desired=[0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0], actual=list(cumsum2))

    @mock.patch('Morpheus.utils.visualization.plt')
    def test_ratio_hist(self, mock_plt):
        plot_ratio_histogram(self.column1, self.column2, bin_edges=np.array(range(0, 20, 2)))
        self.assertTrue(mock_plt.bar.called)
