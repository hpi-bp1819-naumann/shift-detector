import unittest
from unittest.mock import MagicMock

import pandas as pd

from shift_detector.Utils import ColumnType
from shift_detector.preprocessors.Preprocessing import preprocess


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        sales = {'brand': ['Jones LLC', 'Alpha Co', 'Blue Inc', 'Blue Inc', 'Alpha Co',
                           'Jones LLC', 'Alpha Co', 'Blue Inc', 'Blue Inc', 'Alpha Co',
                           'Jones LLC'],
                 'payment': [150, 200, 50, 10, 5, 150, 200, 50, 10, 5, 1],
                 'description': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']}
        self.df1 = pd.DataFrame.from_dict(sales)
        self.df2 = self.df1
        self.columns = list(self.df1.columns.values)

        def mock_preprocessing(i):
            mock = MagicMock()
            mock.process = MagicMock(return_value=i)
            return mock

        self.needed_preprocessings = [mock_preprocessing(i) for i in range(3)]

        needed_preprocessing1 = {
            ColumnType.categorical: self.needed_preprocessings[0],
            ColumnType.text: self.needed_preprocessings[1]
        }

        needed_preprocessing2 = {
            ColumnType.categorical: self.needed_preprocessings[0],
            ColumnType.text: self.needed_preprocessings[2]
        }

        check1 = MagicMock()
        check1.needed_preprocessing = MagicMock(return_value=needed_preprocessing1)

        check2 = MagicMock()
        check2.needed_preprocessing = MagicMock(return_value=needed_preprocessing2)

        self.checks = list([check1, check2])

    def test_preprocessing(self):
        preprocessings = preprocess(self.checks, self.df1, self.df2, self.columns)

        with self.subTest("Test aggregate preprocessings"):
            categorical = preprocessings[ColumnType.categorical]
            self.assertEqual(len(categorical), 1)
            self.assertEqual(categorical[self.needed_preprocessings[0]], 0)

        with self.subTest("Test missing preprocessings"):
            self.assertFalse(ColumnType.numeric in preprocessings)

        with self.subTest("Test accumulation of preprocessings"):
            text = preprocessings[ColumnType.text]
            self.assertEqual(len(text), 2)
            self.assertEqual(text[self.needed_preprocessings[1]], 1)
            self.assertEqual(text[self.needed_preprocessings[2]], 2)
