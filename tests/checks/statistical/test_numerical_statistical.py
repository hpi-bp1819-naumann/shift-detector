import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.detector import Detector
from shift_detector.checks.statistical_checks.numerical_statistical_check import kolmogorov_smirnov_test, \
    NumericalStatisticalCheck
from shift_detector.precalculations.store import Store

import test_data as td


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def test_kolmogorov_smirnov_test_result(self):
        part1 = pd.Series(td.numerical_kolmogorov_smirnov_1)
        part2 = pd.Series(td.numerical_kolmogorov_smirnov_2)
        p = kolmogorov_smirnov_test(part1, part2)
        self.assertAlmostEqual(0.043055, p, places=2)  # this should be equal in 5 places, but travis fails otherwise

    def test_not_significant(self):
        df1 = pd.DataFrame(td.numerical_not_significant_1)
        df2 = pd.DataFrame(td.numerical_not_significant_2)
        store = Store(df1, df2)
        result = NumericalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(0, len(result.shifted_columns))
        self.assertEqual(0, len(result.explanation))

    def test_significant(self):
        df1 = pd.DataFrame(td.numerical_significant_1)
        df2 = pd.DataFrame(td.numerical_significant_2)
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
