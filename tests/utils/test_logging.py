import unittest

import pandas as pd

from shift_detector.Detector import Detector


class TestLogging(unittest.TestCase):

    def test_logging_works_for_unnamed_columns(self):
        df1 = pd.DataFrame(list(range(10)))
        df2 = pd.DataFrame(list(range(11, 20)))
        Detector(df1, df2)  # should not raise exception because columns are unnamed