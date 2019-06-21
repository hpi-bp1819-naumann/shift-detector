import logging as logger
from collections.abc import Iterable
from numbers import Number
from typing import List

from morpheus.checks.check import Check, Report
from morpheus.precalculations.word_prediction_precalculation import WordPredictionPrecalculation
from morpheus.utils.column_management import ColumnType


class WordPredictionCheck(Check):

    def __init__(self, columns=None, ft_window_size=5, ft_size=100, ft_workers=4, ft_seed=None,
                 lstm_window=5, relative_thresh=.8):
        self.columns = columns
        self.relative_thresh = relative_thresh
        self.ft_window_size = ft_window_size
        self.ft_size = ft_size
        self.ft_workers = ft_workers
        self.ft_seed = ft_seed
        self.lstm_window = lstm_window

        if columns and (not isinstance(columns, Iterable) or any(not isinstance(column, str) for column in columns)):
            raise TypeError("columns should be empty or a list of strings. Received: {}".format(columns))

        if not isinstance(self.relative_thresh, Number):
            raise TypeError('Expected argument relative_thresh to be of types [float, int]. '
                            'Received {}.'.format(self.relative_thresh.__class__.__name__))

        if self.relative_thresh <= .0:
            raise ValueError('Expected argument relative_thresh to be >= 0. '
                             'Received {}.'.format(self.relative_thresh))

        if not isinstance(self.lstm_window, int):
            raise TypeError('Expected argument lstm_window to be of type int. '
                            'Received {}.'.format(self.lstm_window.__class__.__name__))

        if self.lstm_window < 1:
            raise ValueError('Expected argument lstm_window to be >= 1 '
                             'Received {}.'.format(self.lstm_window))

    def run(self, store) -> Report:

        if self.columns is None:
            self.columns = store.column_names(ColumnType.text)
            logger.info('Automatically selected columns [{}] to be tested by WordPredictionCheck'.format(self.columns))

        result = {}
        for col in self.columns:
            result[col] = store[WordPredictionPrecalculation(col,
                                                             ft_window_size=self.ft_window_size,
                                                             ft_size=self.ft_size,
                                                             ft_workers=self.ft_workers,
                                                             ft_seed=self.ft_seed,
                                                             lstm_window=self.lstm_window,
                                                             verbose=0)]

        examined_columns = self.columns
        shifted_columns, explanation = self.detect_shifts(examined_columns, result)

        return Report("WordPredictionCheck", examined_columns, shifted_columns, explanation)

    def detect_shifts(self, examined_columns: List[str], result: dict):
        shifted_columns = []
        explanation = {}

        for column in examined_columns:
            df1_prediction_loss, df2_prediction_loss = result[column]
            explanation[column] = "{} -> {}".format(df1_prediction_loss, df2_prediction_loss)
            if df2_prediction_loss > df1_prediction_loss * (1 + self.relative_thresh):
                shifted_columns.append(column)

        return shifted_columns, explanation
