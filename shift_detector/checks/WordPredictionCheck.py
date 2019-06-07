from typing import List

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.WordPredictionPrecalculation import WordPredictionPrecalculation
from shift_detector.utils.ColumnManagement import ColumnType

import logging as logger


class WordPredictionCheck(Check):

    def __init__(self, columns=None, ft_window_size=5, ft_size=100, lstm_window=5, relative_thresh=0.2):
        self.columns = columns
        self.relative_thresh = relative_thresh
        self.ft_window_size = ft_window_size
        self.ft_size = ft_size
        self.lstm_window = lstm_window

        if self.relative_thresh >= .0:
            raise ValueError('Expected argument relative_thresh to be >= 0. '
                             'Received {}.'.format(self.relative_thresh))

    def run(self, store) -> Report:

        if self.columns is None:
            df1_columns = set(store[ColumnType.text][0].columns)
            df2_columns = set(store[ColumnType.text][1].columns)
            self.columns = list(df1_columns.intersection(df2_columns))
            logger.info('Automatically selected columns [{}] to be tested by WordPredictionCheck'.format(self.columns))

        result = {}
        for col in self.columns:
            result[col] = store[WordPredictionPrecalculation(col,
                                                             self.ft_window_size,
                                                             self.ft_size,
                                                             self.lstm_window)]

        examined_columns = self.columns
        shifted_columns, explanation = self.detect_shifts(examined_columns, result)

        return Report(examined_columns, shifted_columns, explanation)

    def detect_shifts(self, examined_columns: List[str], result: dict):
        shifted_columns = []
        explanation = {}

        for column in examined_columns:
            df1_prediction_loss, df2_prediction_loss = result[column]
            explanation[column] = "{} -> {}".format(df1_prediction_loss, df2_prediction_loss)
            if df2_prediction_loss > df1_prediction_loss * (1 + self.relative_thresh):
                shifted_columns.append(column)

        return shifted_columns, explanation
