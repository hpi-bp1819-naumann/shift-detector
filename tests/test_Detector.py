import unittest

import pandas as pd

from shift_detector.Detector import Detector
from shift_detector.Utils import ColumnType
from shift_detector.checks.Chi2Check import Chi2Check


class TestDetector(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ['Jones LLC', 'Alpha Co', 'Blue Inc', 'Blue Inc', 'Alpha Co',
                           'Jones LLC', 'Alpha Co', 'Blue Inc', 'Blue Inc', 'Alpha Co',
                           'Jones LLC'],
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1],
                 'description': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']}
        self.df1 = pd.DataFrame.from_dict(sales)

        self.detector = Detector(df1=self.df1, df2=self.df1)
        self.check = Chi2Check()
        self.detector.add_check(self.check)
        self.detector.run()

    def test_split_dataframes(self):

        splitted_df = self.detector._split_dataframes(self.df1, self.df1,
                                                      columns=["brand", "payment", "description"])
        numeric_columns = splitted_df[ColumnType.numeric][0].columns.values
        categorical_columns = splitted_df[ColumnType.categorical][0].columns.values
        text_columns = splitted_df[ColumnType.text][0].columns.values

        self.assertListEqual(list(numeric_columns), list(['payment']))
        self.assertListEqual(list(categorical_columns), list(['brand', 'payment']))
        self.assertListEqual(list(text_columns), list(['description']))
