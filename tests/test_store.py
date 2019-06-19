import unittest

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from pandas.util.testing import assert_frame_equal

from shift_detector.precalculations.store import InsufficientDataError, Store
from shift_detector.utils.column_management import ColumnType


class TestStore(unittest.TestCase):

    def test_init_custom_column_types(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"] * 10,
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1] * 10,
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] * 10}
        df1 = df2 = pd.DataFrame.from_dict(sales)

        with self.subTest("Successful initialisation"):
            Store(df1, df2, {'description': ColumnType.categorical})

        with self.subTest("Exception when no dict is passed as custom_column_types"):
            self.assertRaises(TypeError, lambda: Store(df1, df2, 'no_dict'))

        with self.subTest("Exception when key of custom_column_types is not a string"):
            self.assertRaises(TypeError, lambda: Store(df1, df2, {0: ColumnType.numerical}))

        with self.subTest("Exception when value of custom_column_types is not a ColumnType"):
            self.assertRaises(TypeError, lambda: Store(df1, df2, {'brand': 0}))

    def test_min_data_size_is_enforced(self):
        df1 = pd.DataFrame(list(range(10)))
        df2 = pd.DataFrame(list(range(10)))
        store = Store(df1=df1, df2=df2)
        assert_frame_equal(df1, store[ColumnType.numerical][0])
        assert_frame_equal(df2, store[ColumnType.numerical][1])
        self.assertRaises(InsufficientDataError, Store, df1=pd.DataFrame(), df2=pd.DataFrame([0]))
        self.assertRaises(InsufficientDataError, Store,
                          df1=pd.DataFrame(list(range(9))),
                          df2=pd.DataFrame(list(range(20))))

    def test_apply_custom_column_types(self):
        data = {'to_numerical': ['150', '200', '50', '10', '5', '150', '200', '50', '10', '5', '1'] * 10,
                'to_text': ['150', '200', '50', '10', '5', '150', '200', '50', '10', '5', '1'] * 10,
                'to_categorical': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1] * 10}
        df1 = df2 = pd.DataFrame.from_dict(data)

        custom_column_types = {
            'to_numerical': ColumnType.numerical,
            'to_text': ColumnType.text,
            'to_categorical': ColumnType.categorical
        }

        store = Store(df1, df2, custom_column_types)

        with self.subTest("Apply custom_column_types"):
            self.assertEqual(['to_categorical'], store.type_to_columns[ColumnType.categorical])
            self.assertEqual(['to_text'], store.type_to_columns[ColumnType.text])
            self.assertEqual(['to_numerical'], store.type_to_columns[ColumnType.numerical])

        with self.subTest("Apply numerical conversion for custom_column_types to dataframes"):
            self.assertTrue(is_numeric_dtype(store.df1['to_numerical']))
            self.assertTrue(store.df1['to_numerical'].equals(pd.Series([150, 200, 50, 10, 5,
                                                                        150, 200, 50, 10, 5, 1] * 10)))

        with self.subTest("Apply categorical conversion for custom_column_types to dataframes"):
            self.assertTrue(is_string_dtype(store.df1['to_categorical']))
            self.assertTrue(store.df1['to_categorical'].equals(pd.Series(['150', '200', '50', '10', '5',
                                                                          '150', '200', '50', '10', '5', '1'] * 10)))

        with self.subTest("Apply textual conversion for custom_column_types to dataframes"):
            self.assertTrue(is_string_dtype(store.df1['to_text']))
            self.assertTrue(store.df1['to_text'].equals(pd.Series(['150', '200', '50', '10', '5',
                                                                          '150', '200', '50', '10', '5', '1'] * 10)))

    def test_change_column_type(self):
        data = {'to_numerical': ['a', '200', '50', '10', '5', '150', '200', '50', '10', '5', '1'] * 10}
        df1 = df2 = pd.DataFrame.from_dict(data)
        custom_column_types = {
            'to_numerical': ColumnType.numerical
        }

        with self.subTest("Exception when trying to convert non-numerical column to numerical"):
            self.assertRaises(Exception, lambda: Store(df1, df2, custom_column_types))

    def test_column_names(self):
        sales = {'brand': ["Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC", "Alpha Co", "Blue Inc", "Blue Inc", "Alpha Co",
                           "Jones LLC"] * 10,
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1] * 10,
                 'description': ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] * 10}
        df1 = pd.DataFrame.from_dict(sales)
        df2 = df1

        store = Store(df1, df2)

        with self.subTest("Test access with column_type"):
            self.assertEqual(['payment'], store.column_names(ColumnType.numerical))
            self.assertCountEqual(['brand', 'description'], store.column_names(ColumnType.categorical, ColumnType.text))

        with self.subTest("Test access without specifying column_type"):
            self.assertCountEqual(['brand', 'payment', 'description'], store.column_names())

        with self.subTest("Incorrect column_types"):
            self.assertRaises(TypeError, lambda: store(ColumnType.numerical, 'no_column_type'))
