import unittest

import pandas as pd

from shift_detector.Utils import ColumnType, shared_column_names, is_categorical, split_dataframes


class TestUtils(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"],
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1],
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]}
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1

    def test_shared_column_names(self):
        with self.subTest():
            shared_columns = shared_column_names(self.df1, self.df2)
            self.assertCountEqual(shared_columns, ['brand', 'payment', 'description'])

        with self.subTest():
            df2_dict = {'brand': ["Jones LLC"],
                        'number': [150],
                        'text': ["A"]}
            df2 = pd.DataFrame.from_dict(df2_dict)

            shared_columns = shared_column_names(self.df1, df2)
            self.assertCountEqual(shared_columns, ['brand'])

        with self.subTest():
            df2_dict = {'number': [150],
                        'text': ["A"]}
            df2 = pd.DataFrame.from_dict(df2_dict)

            self.assertRaises(Exception, shared_column_names, self.df1, df2)

    def test_is_categorical(self):
        self.assertTrue(is_categorical(self.df1['brand']))
        self.assertTrue(is_categorical(self.df1['payment']))
        self.assertFalse(is_categorical(self.df1['payment'], max_unique_fraction=0.05))
        self.assertFalse(is_categorical(self.df1['description']))

    def test_split_dataframes(self):
        splitted_df = split_dataframes(self.df1, self.df2,
                                       columns=['brand', 'payment', 'description'])

        numeric_columns = splitted_df[ColumnType.numerical][0].columns.values
        categorical_columns = splitted_df[ColumnType.categorical][0].columns.values
        text_columns = splitted_df[ColumnType.text][0].columns.values

        self.assertListEqual(list(numeric_columns), list(['payment']))
        self.assertListEqual(list(categorical_columns), list(['brand', 'payment']))
        self.assertListEqual(list(text_columns), list(['description']))
