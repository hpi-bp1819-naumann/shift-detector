import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.Detector import Detector
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test, CategoricalStatisticalCheck
from shift_detector.preprocessors.Store import Store


class TestCategoricalStatisticalCheck(unittest.TestCase):

    def test_chi2_test_result(self):
        part1 = pd.Series((['severe reaction'] * 29) + (['no severe reaction'] * 4757), name='thigh')
        part2 = pd.Series((['severe reaction'] * 75) + (['no severe reaction'] * 8839), name='arm')
        p = chi2_test(part1, part2)
        self.assertAlmostEqual(0.17471089, p)

    def test_not_significant(self):
        df1 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 29) + (['no severe reaction'] * 4757)})
        df2 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 75) + (['no severe reaction'] * 8839)})
        store = Store(df1, df2)
        result = CategoricalStatisticalCheck().run(store)
        self.assertEqual(0, len(result.significant_columns()))

    def test_significant(self):
        df1 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 29) + (['no severe reaction'] * 4757)})
        df2 = pd.DataFrame.from_dict({'vaccination_reaction': (['severe reaction'] * 125) + (['no severe reaction'] * 8839)})
        store = Store(df1, df2)
        result = CategoricalStatisticalCheck().run(store)
        self.assertEqual(1, len(result.significant_columns()))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame([0] * 100)
        df2 = pd.DataFrame([0] * 100)
        detector = Detector(df1=df1, df2=df2)
        detector.add_checks(CategoricalStatisticalCheck())
        detector.run()
        assert_frame_equal(pd.DataFrame([1.0], index=['pvalue']), detector.check_reports[0].result)