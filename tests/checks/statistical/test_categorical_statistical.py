import unittest
from unittest import mock

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.detector import Detector
from shift_detector.checks.statistical_checks.categorical_statistical_check import chi2_test, \
    CategoricalStatisticalCheck
from shift_detector.precalculations.store import Store


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def setUp(self) -> None:
        self.df1_significant = pd.DataFrame.from_dict({'vaccination_reaction':
                                                       (['severe reaction'] * 29) +
                                                       (['no severe reaction'] * 4757)})
        self.df2_significant = pd.DataFrame.from_dict({'vaccination_reaction':
                                                       (['severe reaction'] * 125) +
                                                       (['no severe reaction'] * 8839)})
        self.df1_not_sig = pd.DataFrame.from_dict({'vaccination_reaction':
                                                   (['severe reaction'] * 29) +
                                                   (['no severe reaction'] * 4757)})
        self.df2_not_sig = pd.DataFrame.from_dict({'vaccination_reaction':
                                                   (['severe reaction'] * 75) +
                                                   (['no severe reaction'] * 8839)})

    def test_chi2_test_result(self):
        part1 = pd.Series((['severe reaction'] * 29) + (['no severe reaction'] * 4757), name='thigh')
        part2 = pd.Series((['severe reaction'] * 75) + (['no severe reaction'] * 8839), name='arm')
        p = chi2_test(part1, part2)
        self.assertAlmostEqual(0.17471089, p)

    def test_not_significant(self):
        store = Store(self.df1_not_sig, self.df2_not_sig)
        result = CategoricalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(0, len(result.shifted_columns))
        self.assertEqual(0, len(result.explanation))

    def test_significant(self):
        store = Store(self.df1_significant, self.df2_significant)
        result = CategoricalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(1, len(result.shifted_columns))
        self.assertEqual(1, len(result.explanation))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame([0] * 10)
        df2 = pd.DataFrame([0] * 10)
        detector = Detector(df1=df1, df2=df2, log_print=False)
        detector.run(CategoricalStatisticalCheck())
        self.assertEqual(1, len(detector.check_reports[0].examined_columns))
        self.assertEqual(0, len(detector.check_reports[0].shifted_columns))
        self.assertEqual(0, len(detector.check_reports[0].explanation))
        assert_frame_equal(pd.DataFrame([1.0], index=['pvalue']), detector.check_reports[0].information['test_results'])

    def test_figure_functions_are_collected(self):
        df1 = pd.DataFrame.from_dict(
            {'significant_col': (['severe reaction'] * 29) + (['no severe reaction'] * 4757),
             'not_significant_col': ['uninteresting'] * 4786})
        df2 = pd.DataFrame.from_dict(
            {'significant_col': (['severe reaction'] * 125) + (['no severe reaction'] * 8839),
             'not_significant_col': ['uninteresting'] * 8964})
        check = CategoricalStatisticalCheck()
        result = check.column_figures(significant_columns=['significant_col'], df1=df1, df2=df2)
        self.assertEqual(1, len(result))

    @mock.patch('shift_detector.checks.statistical_checks.categorical_statistical_check.plt')
    def test_paired_total_ratios_figure_looks_right(self, mock_plt):
        with mock.patch.object(pd.DataFrame, 'plot') as mock_plot:
            CategoricalStatisticalCheck.paired_total_ratios_figure('vaccination_reaction',
                                                                   self.df1_significant, self.df2_significant)
        self.assertTrue(mock_plot.called)
        self.assertTrue(mock_plot.return_value.set_title.called)
        self.assertTrue(mock_plot.return_value.set_xlabel.called)
        self.assertTrue(mock_plot.return_value.set_ylabel.called)
        self.assertTrue(mock_plot.return_value.invert_yaxis.called)
        self.assertTrue(mock_plt.show.called)
