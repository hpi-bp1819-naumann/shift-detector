import unittest
from unittest import mock

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.checks.statistical_checks import numerical_statistical_check
from shift_detector.checks.statistical_checks.numerical_statistical_check import kolmogorov_smirnov_test, \
    NumericalStatisticalCheck
from shift_detector.detector import Detector
from shift_detector.precalculations.store import Store


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def setUp(self):
        self.not_significant_1 = ([2] * 1) +\
                                 ([3] * 2) +\
                                 ([4] * 0) +\
                                 ([5] * 1) +\
                                 ([6] * 2) +\
                                 ([8] * 3) +\
                                 ([9] * 2)

        self.not_significant_2 = ([2] * 2) +\
                                 ([3] * 1) +\
                                 ([4] * 2) +\
                                 ([5] * 2) +\
                                 ([6] * 1) +\
                                 ([8] * 1) +\
                                 ([9] * 1)

        self.kolmogorov_smirnov_1 = ([21] * 4) +\
                                    ([23] * 11) +\
                                    ([25] * 5) +\
                                    ([27] * 7) +\
                                    ([29] * 0) +\
                                    ([31] * 5) +\
                                    ([33] * 9) +\
                                    ([35] * 13) +\
                                    ([37] * 20) +\
                                    ([39] * 6)

        self.kolmogorov_smirnov_2 = ([21] * 7) +\
                                    ([23] * 4) +\
                                    ([25] * 1) +\
                                    ([27] * 11) +\
                                    ([29] * 12) +\
                                    ([31] * 4) +\
                                    ([33] * 2) +\
                                    ([35] * 4) +\
                                    ([37] * 8) +\
                                    ([39] * 9)

        self.significant_1 = ([21] * 7) +\
                             ([23] * 4) +\
                             ([25] * 1) +\
                             ([27] * 11) +\
                             ([29] * 12) +\
                             ([31] * 4) +\
                             ([33] * 2) +\
                             ([35] * 4) +\
                             ([37] * 2) +\
                             ([39] * 9)

        self.significant_2 = ([21] * 4) +\
                             ([23] * 11) +\
                             ([25] * 5) +\
                             ([27] * 7) +\
                             ([29] * 0) +\
                             ([31] * 5) +\
                             ([33] * 9) +\
                             ([35] * 18) +\
                             ([37] * 29) +\
                             ([39] * 6)

    def test_kolmogorov_smirnov_test_result(self):
        part1 = pd.Series(self.kolmogorov_smirnov_1)
        part2 = pd.Series(self.kolmogorov_smirnov_2)
        p = kolmogorov_smirnov_test(part1, part2)
        self.assertAlmostEqual(0.043055, p, places=2)  # this should be equal in 5 places, but travis fails otherwise

    def test_not_significant(self):
        df1 = pd.DataFrame(self.not_significant_1)
        df2 = pd.DataFrame(self.not_significant_2)
        store = Store(df1, df2)
        result = NumericalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(0, len(result.shifted_columns))
        self.assertEqual(0, len(result.explanation))

    def test_significant(self):
        df1 = pd.DataFrame(self.significant_1)
        df2 = pd.DataFrame(self.significant_2)
        store = Store(df1, df2)
        result = NumericalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(1, len(result.shifted_columns))
        self.assertEqual(1, len(result.explanation))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame([0] * 10)
        df2 = pd.DataFrame([0] * 10)
        detector = Detector(df1=df1, df2=df2)
        detector.run(NumericalStatisticalCheck())
        self.assertEqual(1, len(detector.check_reports[0].examined_columns))
        self.assertEqual(0, len(detector.check_reports[0].shifted_columns))
        self.assertEqual(0, len(detector.check_reports[0].explanation))
        assert_frame_equal(pd.DataFrame([1.0], index=['pvalue']), detector.check_reports[0].information['test_results'])

    def test_figure_functions_are_collected(self):
        df1 = pd.DataFrame(self.significant_1)
        df2 = pd.DataFrame(self.significant_2)
        df1['non_sig_col'] = [0] * len(df1)
        df2['non_sig_col'] = [0] * len(df2)
        check = NumericalStatisticalCheck()
        result = check.column_figures(significant_columns=['significant_col'],
                                      df1=df1, df2=df2)
        self.assertEqual(1, len(result))

    @mock.patch('shift_detector.checks.statistical_checks.numerical_statistical_check.plt')
    def test_cumulative_hist_figure_looks_right(self, mock_plt):
        df1 = pd.DataFrame(self.significant_1, columns=['meaningful_numbers'])
        df2 = pd.DataFrame(self.significant_2, columns=['meaningful_numbers'])
        with mock.patch.object(numerical_statistical_check.vis, 'plot_cumulative_step_ratio_histogram',
                               return_value=[np.array([0, 1, 2, 3]),
                                             np.array([0, 1, 2, 3])]) as mock_cum_hist:
            NumericalStatisticalCheck.cumulative_hist_figure('meaningful_numbers',
                                                             df1, df2, bins=3)
        self.assertTrue(mock_cum_hist.called)
        self.assertTrue(mock_plt.plot.called)
        self.assertTrue(mock_plt.legend.called)
        self.assertTrue(mock_plt.title.called)
        self.assertTrue(mock_plt.xlabel.called)
        self.assertTrue(mock_plt.ylabel.called)
        self.assertTrue(mock_plt.show.called)

    @mock.patch('shift_detector.checks.statistical_checks.numerical_statistical_check.plt')
    def test_overlayed_hist_figure_looks_right(self, mock_plt):
        df1 = pd.DataFrame(self.significant_1, columns=['meaningful_numbers'])
        df2 = pd.DataFrame(self.significant_2, columns=['meaningful_numbers'])
        with mock.patch.object(numerical_statistical_check.vis, 'plot_ratio_histogram') as mock_ratio_hist:
            NumericalStatisticalCheck.overlayed_hist_figure('meaningful_numbers',
                                                            df1, df2, bins=3)
        self.assertTrue(mock_ratio_hist.called)
        self.assertTrue(mock_plt.legend.called)
        self.assertTrue(mock_plt.title.called)
        self.assertTrue(mock_plt.xlabel.called)
        self.assertTrue(mock_plt.ylabel.called)
        self.assertTrue(mock_plt.show.called)
