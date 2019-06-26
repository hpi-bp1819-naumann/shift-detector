import logging as logger
from collections.abc import Iterable
from numbers import Number
from typing import List

import pandas as pd
from IPython.core.display import display

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.word_prediction_precalculation import WordPredictionPrecalculation
from shift_detector.utils.column_management import ColumnType


class WordPredictionCheck(Check):

    def __init__(self, columns=None, ft_window_size=5, ft_size=100, ft_workers=4, seed=None,
                 lstm_window=5, relative_thresh=.15,
                 output_path="wordPredictionCheck_model_checkpoints"):
        self.columns = columns
        self.relative_thresh = relative_thresh
        self.ft_window_size = ft_window_size
        self.ft_size = ft_size
        self.ft_workers = ft_workers
        self.seed = seed
        self.lstm_window = lstm_window
        self.output_path = output_path

        if columns and (not isinstance(columns, Iterable) or
                        any(not isinstance(column, str) for column in columns)):
            raise TypeError("columns should be empty or "
                            "a list of strings. Received: {}".format(columns))

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
            logger.info('Automatically selected columns [{}] '
                        'to be tested by WordPredictionCheck'.format(self.columns))

        result = {}
        failure_columns = {}
        for col in self.columns:
            try:
                result[col] = store[WordPredictionPrecalculation(col,
                                                                 ft_window_size=self.ft_window_size,
                                                                 ft_size=self.ft_size,
                                                                 ft_workers=self.ft_workers,
                                                                 seed=self.seed,
                                                                 lstm_window=self.lstm_window,
                                                                 verbose=0,
                                                                 output_path=self.output_path)]
            except ValueError as e:
                failure_columns[col] = e
                logger.warning('Skipping textual column {} '
                               'due to the following error: {}'.format(col, e))

        examined_columns = self.columns
        columns_no_failure = list(set(examined_columns) - failure_columns.keys())
        shifted_columns, explanation = self.detect_shifts(columns_no_failure, result)

        information = {col: err for col, err in failure_columns.items()}

        return WordPredictionReport("Word Prediction Check", examined_columns, shifted_columns,
                                    explanation, information)

    def detect_shifts(self, examined_columns: List[str], result: dict):
        shifted_columns = []
        COL, LOSS_DF1, LOSS_DF2, ABS_DIFF, REL_DIFF, REL_THRESH = 'column', 'loss on df1',\
                                                                  'loss on df2', \
                                                                  'absolute difference', \
                                                                  'relative difference', \
                                                                  'relative threshold'
        data = {
            COL: [],
            LOSS_DF1: [],
            LOSS_DF2: [],
            ABS_DIFF: [],
            REL_DIFF: [],
            REL_THRESH: []
        }

        for column in examined_columns:
            temp_data = dict()

            temp_data[COL] = column
            temp_data[LOSS_DF1], temp_data[LOSS_DF2] = result[column]
            temp_data[ABS_DIFF] = temp_data[LOSS_DF2] - temp_data[LOSS_DF1]
            temp_data[REL_DIFF] = temp_data[ABS_DIFF] / temp_data[LOSS_DF1]

            if temp_data[REL_DIFF] > self.relative_thresh:
                shifted_columns.append(column)

            temp_data[REL_DIFF] = self.__beautify_percentage(temp_data[REL_DIFF])
            temp_data[REL_THRESH] = self.__beautify_percentage(self.relative_thresh)

            for k, v in temp_data.items():
                data[k] += [v]

        return shifted_columns, pd.DataFrame(data=data)

    @staticmethod
    def __beautify_percentage(num):
        sign = '+' if num >= 0 else ''
        return sign + "{:.2f}%".format(num)


class WordPredictionReport(Report):

    def print_explanation(self):
        print("Results per column:")

        self.explanation = self.explanation.style.apply(
            lambda x: ['background: #FF6A6A' if x['column'] in self.shifted_columns else ''] * len(x),
            axis=1)

        display(self.explanation)
