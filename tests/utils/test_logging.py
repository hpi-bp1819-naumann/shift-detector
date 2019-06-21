import unittest

import pandas as pd

from Morpheus.detector import Detector


class TestLogging(unittest.TestCase):

    def test_logging_works_for_unnamed_columns(self):
        df1 = pd.DataFrame(list(range(11)))
        df2 = pd.DataFrame(list(range(11, 21)))
        Detector(df1, df2)  # should not raise exception because columns are unnamed
