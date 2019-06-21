import logging as logger
import os
from unittest import TestCase

from numpy.random import seed
from pandas import DataFrame
from tensorflow import set_random_seed

from Morpheus.precalculations.store import Store
from Morpheus.precalculations.word_prediction_precalculation import WordPredictionPrecalculation


class TestWordPredictionPrecalculation(TestCase):

    def setUp(self):

        # set seeds
        seed(1)
        set_random_seed(1)
        os.environ['PYTHONHASHSEED'] = "0"

        # create alphabet list: ['a', 'b', ..., 'z']
        alphabet = [chr(letter) for letter in range(ord('a'), ord('z')+1)]

        col = []
        # creates lists of size 7 with consecutive letters from the alphabet
        # result: ['a b c d e f g', 'b c d e f g h', ...]
        for idx in range(len(alphabet) - 7):
            col.append(' '.join(alphabet[idx:idx+7]))

        error_col = []
        for idx in range(len(col)):
            error_col += [alphabet[idx]]

        data1 = {'shift': col, 'no_shift': col, 'error_col': error_col}
        data2 = {'shift': ['B B B B B B'] * len(col), 'no_shift': col, 'error_col': error_col}
        self.df1 = DataFrame.from_dict(data1)
        self.df2 = DataFrame.from_dict(data2)
        self.store = Store(self.df1, self.df2)
        self.precalculation_shift = WordPredictionPrecalculation('shift', num_epochs_predictor=10,
                                                                 ft_workers=2, ft_seed=0)
        self.precalculation_no_shift = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10,
                                                                    ft_workers=1, ft_seed=0)
        self.precalculation_error = WordPredictionPrecalculation('error_col', num_epochs_predictor=10,
                                                                 ft_workers=1, ft_seed=0)

    def test_process(self):
        with self.subTest("Test losses"):
            df1_prediction_loss, df2_prediction_loss = self.precalculation_shift.process(self.store)
            min_loss_diff = df1_prediction_loss * .15
            logger.info('prediction_loss in first dataset: ', df1_prediction_loss)
            logger.info('prediction_loss in second dataset: ', df2_prediction_loss)
            self.assertTrue(df2_prediction_loss > df1_prediction_loss + min_loss_diff)

            df1_prediction_loss, df2_prediction_loss = self.precalculation_no_shift.process(self.store)
            self.assertTrue(df2_prediction_loss <= df1_prediction_loss)

    def test_error_raising(self):
        self.assertRaises(ValueError, lambda: self.precalculation_error.process(self.store))

    def test_equal(self):
        with self.subTest("Test equality"):
            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10, ft_seed=0)
            self.assertEqual(self.precalculation_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10, ft_seed=0)
            self.assertEqual(self.precalculation_no_shift, other_precalculation)

        with self.subTest("Test inequality"):
            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10, ft_seed=1)
            self.assertNotEqual(self.precalculation_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10)
            self.assertNotEqual(self.precalculation_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10, ft_seed=1)
            self.assertNotEqual(self.precalculation_no_shift, other_precalculation)

            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10)
            self.assertNotEqual(self.precalculation_no_shift, other_precalculation)

        with self.subTest("Test inequality with another class"):
            other = "Not a WordPredictionPrecalculation"
            self.assertNotEqual(self.precalculation_shift, other)
            self.assertNotEqual(self.precalculation_no_shift, other)

    def test_hash(self):
        with self.subTest("Test hash equality"):
            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10, ft_seed=0)
            self.assertEqual(self.precalculation_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10, ft_seed=0)
            self.assertEqual(self.precalculation_no_shift.__hash__(), other_precalculation.__hash__())

        with self.subTest("Test hash inequality"):
            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10)
            self.assertNotEqual(self.precalculation_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10, ft_seed=1)
            self.assertNotEqual(self.precalculation_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('shift', num_epochs_predictor=10)
            self.assertNotEqual(self.precalculation_no_shift.__hash__(), other_precalculation.__hash__())

            other_precalculation = WordPredictionPrecalculation('no_shift', num_epochs_predictor=10, ft_seed=1)
            self.assertNotEqual(self.precalculation_no_shift.__hash__(), other_precalculation.__hash__())
