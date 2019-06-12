import logging as logger
from collections.abc import Iterable
from numbers import Number

from sklearn.metrics import classification_report

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.distinction_precalculation import DistinctionPrecalculation


class DistinctionCheck(Check):

    def __init__(self, columns=[], num_epochs=10, relative_threshold=.1):
        if columns and (not isinstance(columns, Iterable) or any(not isinstance(column, str) for column in columns)):
            raise TypeError("columns should be empty or a list of strings. Received: {}".format(columns))
        self.columns = columns

        if not isinstance(num_epochs, int):
            raise TypeError("num_epochs should be an Integer. Received: {} ({})".format(num_epochs,
                                                                                        num_epochs.__class__.__name__))
        if num_epochs < 1:
            raise ValueError("num_epochs should be greater than 0. Received: {}.".format(num_epochs))
        self.num_epochs = num_epochs

        if not isinstance(relative_threshold, Number):
            raise TypeError("relative_threshold should be a Number. "
                            "Received: {} ({})".format(relative_threshold, relative_threshold.__class__.__name__))
        if not 0 <= relative_threshold <= 1:
            raise ValueError("relative_threshold should be greater equal than 0 and smaller equal 1. "
                             "Received: {}.".format(relative_threshold))
        self.relative_threshold = float(relative_threshold)

    def run(self, store) -> Report:
        logger.info("Execute Distinction Check")
        input_columns = self.columns
        if not input_columns:
            input_columns = store.column_names()

        examined_columns, precalculation_result = store[DistinctionPrecalculation(input_columns, self.num_epochs)]

        shifted_columns, explanation = self.detect_shifts(examined_columns, precalculation_result)
        information = self.information(precalculation_result)

        return Report("Distinction Check", examined_columns, shifted_columns, explanation, information)

    def detect_shifts(self, examined_columns, result):
        shifted_columns = []
        explanation = {}

        base_accuracy = result['base_accuracy']
        permuted_accuracies = result['permuted_accuracies']

        for column in examined_columns:
            accuracy = permuted_accuracies[column]
            explanation[column] = "{} -> {}".format(base_accuracy, accuracy)
            if accuracy < base_accuracy * (1 - self.relative_threshold):
                shifted_columns.append(column)

        return shifted_columns, explanation

    @staticmethod
    def information(precalculation_result):
        y_true = precalculation_result['y_true']
        y_pred = precalculation_result['y_pred']

        report = classification_report(y_true, y_pred)
        classification = classification_report(y_true, y_pred, output_dict=True)

        information = dict()
        information['Classification Report'] = report
        information['F1 score df1'] = classification["A"]["f1-score"]
        information['F1 score df2'] = classification["B"]["f1-score"]
        return information
