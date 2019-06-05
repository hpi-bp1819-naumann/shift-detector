import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.Detector import Detector
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test, \
    NumericalStatisticalCheck
from shift_detector.precalculations.Store import Store


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def test_kolmogorov_smirnov_test_result(self):
        part1 = pd.Series(([21] * 4) +
                          ([23] * 11) +
                          ([25] * 5) +
                          ([27] * 7) +
                          ([29] * 0) +
                          ([31] * 5) +
                          ([33] * 9) +
                          ([35] * 13) +
                          ([37] * 20) +
                          ([39] * 6))
        part2 = pd.Series(([21] * 7) +
                          ([23] * 4) +
                          ([25] * 1) +
                          ([27] * 11) +
                          ([29] * 12) +
                          ([31] * 4) +
                          ([33] * 2) +
                          ([35] * 4) +
                          ([37] * 8) +
                          ([39] * 9))
        p = kolmogorov_smirnov_test(part1, part2)
        self.assertAlmostEqual(0.043055, p, places=2)  # this should be equal in 5 places, but travis fails otherwise

    def test_not_significant(self):
        df1 = pd.DataFrame(([2] * 1) +
                           ([3] * 2) +
                           ([4] * 0) +
                           ([5] * 1) +
                           ([6] * 0) +
                           ([8] * 3) +
                           ([9] * 1))
        df2 = pd.DataFrame(([2] * 2) +
                           ([3] * 1) +
                           ([4] * 2) +
                           ([5] * 0) +
                           ([6] * 1) +
                           ([8] * 1) +
                           ([9] * 0))
        store = Store(df1, df2)
        result = NumericalStatisticalCheck().run(store)
        self.assertEqual(0, len(result.significant_columns()))

    def test_significant(self):
        df1 = pd.DataFrame(([21] * 4) +
                           ([23] * 11) +
                           ([25] * 5) +
                           ([27] * 7) +
                           ([29] * 0) +
                           ([31] * 5) +
                           ([33] * 9) +
                           ([35] * 18) +
                           ([37] * 29) +
                           ([39] * 6))
        df2 = pd.DataFrame(([21] * 7) +
                           ([23] * 4) +
                           ([25] * 1) +
                           ([27] * 11) +
                           ([29] * 12) +
                           ([31] * 4) +
                           ([33] * 2) +
                           ([35] * 4) +
                           ([37] * 2) +
                           ([39] * 9))
        store = Store(df1, df2)
        result = NumericalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.significant_columns()))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame([0])
        df2 = pd.DataFrame([0])
        detector = Detector(df1=df1, df2=df2)
        detector.run(NumericalStatisticalCheck())
        assert_frame_equal(pd.DataFrame([1.0], index=['pvalue']), detector.check_reports[0].result)
