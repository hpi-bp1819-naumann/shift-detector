from unittest import TestCase

import numpy as np
from pandas import DataFrame

from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.WordPredictionPrecalculation import WordPredictionPrecalculation


class TestCreateDetector(TestCase):

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
        self.precalculation_shift = WordPredictionPrecalculation('shift')
        self.precalculation_noShift = WordPredictionPrecalculation('no_shift')

        np.random.seed(1)

    def test_process(self):
        with self.subTest("Test losses"):
            df1_prediction_loss, df2_prediction_loss = self.precalculation_shift.process(self.store)
            self.assertTrue(df2_prediction_loss > df1_prediction_loss * 1.3)

            df1_prediction_loss, df2_prediction_loss = self.precalculation_noShift.process(self.store)
            self.assertTrue(df2_prediction_loss <= df1_prediction_loss)

    def test_equal(self):
        with self.subTest("Test equality"):
            other_precalculation = WordPredictionPrecalculation('shift')
            self.assertEqual(self.precalculation_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('no_shift')
            self.assertEqual(self.precalculation_noShift, other_precalculation)

        with self.subTest("Test inequality"):
            other_precalculation = WordPredictionPrecalculation('no_shift')
            self.assertNotEqual(self.precalculation_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('shift')
            self.assertNotEqual(self.precalculation_noShift, other_precalculation)

        with self.subTest("Test inequality with another class"):
            other = "Not a WordPredictionPrecalculation"
            self.assertNotEqual(self.precalculation_shift, other)
            self.assertNotEqual(self.precalculation_noShift, other)

    def test_hash(self):
        with self.subTest("Test hash equality"):
            other_precalculation = WordPredictionPrecalculation('shift')
            self.assertEqual(self.precalculation_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('no_shift')
            self.assertEqual(self.precalculation_noShift.__hash__(), other_precalculation.__hash__())

        with self.subTest("Test hash inequality"):
            other_precalculation = WordPredictionPrecalculation('no_shift')
            self.assertNotEqual(self.precalculation_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('shift')
            self.assertNotEqual(self.precalculation_noShift.__hash__(), other_precalculation.__hash__())
