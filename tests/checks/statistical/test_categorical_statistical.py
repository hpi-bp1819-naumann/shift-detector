import unittest

import pandas as pd

from shift_detector.Detector import Detector
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test, CategoricalStatisticalCheck


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def setUp(self):
        pass

    def test_chi2_test_result(self):
        part1 = pd.Series((['severe reaction'] * 29) + (['no severe reaction'] * 4757), name='thigh')
        part2 = pd.Series((['severe reaction'] * 75) + (['no severe reaction'] * 8839), name='arm')
        p = chi2_test(part1, part2)
        self.assertAlmostEqual(p, 0.17471089)

    def test_check_result(self):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        detector = Detector(df1=df1,df2=df2).add_check(CategoricalStatisticalCheck()).run()
        detector.checks_reports

    def tearDown(self):
        pass