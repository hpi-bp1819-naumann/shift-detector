import os
from unittest import TestCase

from pandas import DataFrame

from shift_detector.checks.word_prediction_check import WordPredictionCheck
from shift_detector.precalculations.store import Store


class TestWordPredictionCheck(TestCase):

    def setUp(self):

        # set hash seed
        os.environ['PYTHONHASHSEED'] = "0"

        alphabet = [chr(letter) for letter in range(ord('a'), ord('z')+1)]

        col = []
        for idx in range(len(alphabet) - 7):
            col.append(' '.join(alphabet[idx:idx+7]))

        col_too_short = [alphabet[i] for i in range(len(col))]

        data1 = {'shift': col, 'no_shift': col, 'too_short': col_too_short}
        data2 = {'shift': ['B B B B B B'] * len(col), 'no_shift': col, 'too_short': col_too_short}
        self.df1 = DataFrame.from_dict(data1)
        self.df2 = DataFrame.from_dict(data2)
        self.store = Store(self.df1, self.df2)
        self.check_automatic_col_detection = WordPredictionCheck(relative_thresh=.15,
                                                                 ft_size=10, ft_workers=1, seed=1)
        self.check_custom_cols = WordPredictionCheck(columns=['shift', 'no_shift'],
                                                     relative_thresh=.15,
                                                     ft_size=10, ft_workers=1, seed=1)

    def test_init(self):
        with self.subTest("Test wrong columns"):
            self.assertRaises(TypeError, lambda: WordPredictionCheck(['shift', 0]))

        with self.subTest("Test wrong relative_thresh"):
            self.assertRaises(TypeError, lambda: WordPredictionCheck(relative_thresh='wrong'))
            self.assertRaises(ValueError, lambda: WordPredictionCheck(relative_thresh=-1))

        with self.subTest("Test wrong lstm_window"):
            self.assertRaises(TypeError, lambda: WordPredictionCheck(lstm_window='wrong'))
            self.assertRaises(TypeError, lambda: WordPredictionCheck(lstm_window=.9))
            self.assertRaises(ValueError, lambda: WordPredictionCheck(lstm_window=0))

    def test_run(self):
        report_automatic_col_detection = self.check_automatic_col_detection.run(self.store)
        report_custom_cols = self.check_custom_cols.run(self.store)

        with self.subTest("Test setting columns"):
            self.assertCountEqual(['shift', 'no_shift'], report_custom_cols.examined_columns)

        with self.subTest("Test shifted columns"):
            self.assertCountEqual(['shift'], report_automatic_col_detection.shifted_columns)

        with self.subTest("Test examined columns"):
            self.assertCountEqual(['shift', 'no_shift', 'too_short'],
                                  report_automatic_col_detection.examined_columns)

        with self.subTest("Test failure columns"):
            self.assertTrue('too_short' in report_automatic_col_detection.information.keys())
            self.assertIsInstance(report_automatic_col_detection.information['too_short'],
                                  ValueError)
