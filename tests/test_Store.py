import unittest

import pandas as pd

from shift_detector.precalculations.Store import InsufficientDataError, Store


class TestStore(unittest.TestCase):

    def test_min_data_size_is_enforced(self):
        self.assertRaises(InsufficientDataError, Store, df1=pd.DataFrame([0]), df2=pd.DataFrame([0]))