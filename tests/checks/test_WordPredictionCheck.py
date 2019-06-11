from unittest import TestCase

from pandas import DataFrame

from shift_detector.checks.WordPredictionCheck import WordPredictionCheck
from shift_detector.precalculations.Store import Store


class TestDistinctionCheck(TestCase):

    def setUp(self):
        alphabet = [chr(letter) for letter in range(ord('a'), ord('z')+1)]

        col = []
        for idx in range(len(alphabet) - 7):
            col.append(' '.join(alphabet[idx:idx+7]))

        data1 = {'shift': col, 'no_shift': col}
        data2 = {'shift': ['B B B B B B'] * len(col), 'no_shift': col}
        self.df1 = DataFrame.from_dict(data1)
        self.df2 = DataFrame.from_dict(data2)
        self.store = Store(self.df1, self.df2)
        self.precalculation = WordPredictionCheck(columns=['shift', 'no_shift'], relative_thresh=.3)

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
        report = self.precalculation.run(self.store)

        with self.subTest("Test shifted columns"):
            self.assertCountEqual(report.shifted_columns, ['shift'])

        with self.subTest("Test examined columns"):
            self.assertCountEqual(report.examined_columns, ['shift', 'no_shift'])
