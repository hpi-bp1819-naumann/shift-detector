import unittest

import numpy as np
import pandas as pd

from morpheus.utils.column_management import is_categorical, detect_column_types, ColumnType, is_binary
from morpheus.utils.data_io import shared_column_names
from morpheus.utils.ucb_list import block, blocks


class TestUtils(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"] * 10,
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1] * 10,
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] * 10,
                 'delivered': [True, False, True, False, True, False, True, False, True, False, True] * 10
                 }
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1

    def test_shared_column_names(self):
        with self.subTest():
            shared_columns = shared_column_names(self.df1, self.df2)
            self.assertCountEqual(shared_columns, ['brand', 'payment', 'description', 'delivered'])

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

    def test_is_binary(self):
        data = {
            'bool': [True, False, True] * 10,
            'bool_with_na': [True, False, np.nan] * 10,
            'no_bool': ['A', 'B', 'C'] * 10
        }
        df = pd.DataFrame.from_dict(data)

        with self.subTest("Test detection of boolean"):
            self.assertTrue(is_binary(df['bool']))

        with self.subTest("Test detection of boolean with nan values"):
            self.assertTrue(is_binary(df['bool_with_na']))
            self.assertFalse(is_binary(df['bool_with_na'], allow_na=False))

        with self.subTest("Test no binary detection for no boolean value"):
            self.assertFalse(is_binary(df['no_bool']))

    def test_detect_column_types(self):
        column_type_to_column_names = detect_column_types(self.df1, self.df2,
                                                          columns=['brand', 'payment', 'description', 'delivered'])

        numerical_columns = column_type_to_column_names[ColumnType.numerical]
        categorical_columns = column_type_to_column_names[ColumnType.categorical]
        text_columns = column_type_to_column_names[ColumnType.text]

        self.assertCountEqual(['payment'], numerical_columns)
        self.assertCountEqual(['brand', 'delivered'], categorical_columns)
        self.assertCountEqual(['description'], text_columns)

    def test_ucblist_block_function(self):
        self.assertEqual('Basic Latin', block('L'))
        self.assertEqual('CJK Unified Ideographs', block('中'))

    def test_ucblist_blocks_dict(self):
        self.assertEqual(300, len(blocks))
        self.assertEqual((0, 127, 'Basic Latin'), blocks[0])
        self.assertEqual((1048576, 1114111, 'Supplementary Private Use Area-B'), blocks[-1])
