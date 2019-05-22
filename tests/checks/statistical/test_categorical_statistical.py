import unittest

import pandas as pd

from shift_detector.Detector import Detector
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test, CategoricalStatisticalCheck


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def test_chi2_test_result(self):
        part1 = pd.Series((['severe reaction'] * 29) + (['no severe reaction'] * 4757), name='thigh')
        part2 = pd.Series((['severe reaction'] * 75) + (['no severe reaction'] * 8839), name='arm')
        p = chi2_test(part1, part2)
        self.assertAlmostEqual(0.17471089, p)

    def test_not_significant(self):
        df1 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 29) + (['no severe reaction'] * 4757)})
        df2 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 75) + (['no severe reaction'] * 8839)})
        detector = Detector(df1=df1, df2=df2)
        detector.add_checks(CategoricalStatisticalCheck())
        detector.run()
        self.assertEqual(0, len(detector.check_reports[0].significant_columns()))

    def test_significant(self):
        df1 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 29) + (['no severe reaction'] * 4757)})
        df2 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 125) + (['no severe reaction'] * 8839)})
        detector = Detector(df1=df1, df2=df2)
        detector.add_checks(CategoricalStatisticalCheck())
        detector.run()
        self.assertEqual(1, len(detector.check_reports[0].significant_columns()))
