import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.precalculations.store import InsufficientDataError, Store
from shift_detector.utils.column_management import ColumnType


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
