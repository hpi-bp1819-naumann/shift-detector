import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.precalculations.Store import InsufficientDataError, Store
from shift_detector.utils.ColumnManagement import ColumnType


class TestStore(unittest.TestCase):

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
            self.assertRaises(TypeError, lambda: store('no_column_type'))
