import unittest

import pandas as pd

from shift_detector.utils.ColumnManagement import is_categorical, detect_column_types, ColumnType
from shift_detector.utils.DataIO import shared_column_names
from shift_detector.utils.UCBlist import block, blocks


class TestUtils(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"] * 10,
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1] * 10,
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] * 10}
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
        column_type_to_column_names = detect_column_types(self.df1, self.df2,
                                                          columns=['brand', 'payment', 'description'])

        numerical_columns = column_type_to_column_names[ColumnType.numerical]
        categorical_columns = column_type_to_column_names[ColumnType.categorical]
        low_cardinal_numerical_columns = column_type_to_column_names[ColumnType.low_cardinal_numerical]
        text_columns = column_type_to_column_names[ColumnType.text]

        self.assertCountEqual(['payment'], numerical_columns)
        self.assertCountEqual(['brand'], categorical_columns)
        self.assertCountEqual(['payment'], low_cardinal_numerical_columns)
        self.assertCountEqual(['description'], text_columns)

    def test_ucblist_block_function(self):
        self.assertEqual('Basic Latin', block('L'))
        self.assertEqual('CJK Unified Ideographs', block('中'))

    def test_ucblist_blocks_dict(self):
        self.assertEqual(300, len(blocks))
        self.assertEqual((0, 127, 'Basic Latin'), blocks[0])
        self.assertEqual((1048576, 1114111, 'Supplementary Private Use Area-B'), blocks[-1])
