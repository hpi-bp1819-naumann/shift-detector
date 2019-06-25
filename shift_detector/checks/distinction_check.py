from collections.abc import Iterable
from numbers import Number

import numpy as np
from IPython.display import display_html
from pandas import DataFrame
from sklearn.metrics import precision_recall_fscore_support

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.distinction_precalculation import DistinctionPrecalculation
from shift_detector.utils.custom_print import nprint, is_in_jupyter


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
        input_columns = self.columns
        if not input_columns:
            input_columns = store.column_names()
        input_columns = set(input_columns)

        complete_examined_columns = []
        complete_shifted_columns = []
        complete_explanation = []
        complete_information = []

        while input_columns:
            examined_columns, precalculation_result = store[DistinctionPrecalculation(list(input_columns),
                                                                                      self.num_epochs)]
            if not complete_examined_columns:
                complete_examined_columns = examined_columns.copy()

            shifted_columns, explanation = self.detect_shifts(examined_columns, precalculation_result)

            complete_shifted_columns.extend(shifted_columns)
            complete_explanation.append(explanation)
            information = self.information(precalculation_result)
            complete_information.append(information)

            if not shifted_columns:
                break
            input_columns -= set(shifted_columns)

        return DistinctionReport("Distinction Check",
                                 complete_examined_columns,
                                 complete_shifted_columns,
                                 complete_explanation,
                                 complete_information)

    def detect_shifts(self, examined_columns, result):
        shifted_columns = []

        base_accuracy = result['base_accuracy']
        permuted_accuracies = result['permuted_accuracies']

        explanation = DataFrame(columns=['column', 'base accuracy in %', 'accuracy in %'])

        relative_accuracy = base_accuracy * (1 - self.relative_threshold)
        for i, column in enumerate(examined_columns):
            accuracy = permuted_accuracies[column]
            explanation.loc[i] = [column, base_accuracy * 100, accuracy * 100]
            if accuracy < relative_accuracy:
                shifted_columns.append(column)

        explanation.loc[:, 'diff'] = explanation['base accuracy in %'] - explanation['accuracy in %']
        explanation = explanation.round(2)
        explanation = explanation.sort_values(by=['diff'], ascending=False).reset_index(drop=True)

        explanation = {'data': explanation, 'relative_accuracy': relative_accuracy * 100}
        return shifted_columns, explanation

    @staticmethod
    def information(precalculation_result):
        y_true = precalculation_result['y_true']
        y_pred = precalculation_result['y_pred']

        total = str(len(y_true))

        df = DataFrame(columns=['label', 'precision', 'recall', 'fscore', 'support'])
        standard_report = np.transpose(precision_recall_fscore_support(y_true, y_pred, labels=['A', 'B']))
        df.loc[0] = ['A'] + list(standard_report[0])
        df.loc[1] = ['B'] + list(standard_report[1])
        micro = list(precision_recall_fscore_support(y_true, y_pred, average='micro'))[:3]
        df.loc[2] = ['micro avg'] + micro + [total]
        macro = list(precision_recall_fscore_support(y_true, y_pred, average='macro'))[:3]
        df.loc[3] = ['macro avg'] + macro + [total]
        weighted = list(precision_recall_fscore_support(y_true, y_pred, average='weighted'))[:3]
        df.loc[4] = ['weighted avg'] + weighted + [total]
        df = df.round(4)

        return df


class DistinctionReport(Report):

    def print_explanation(self):
        for i in range(len(self.explanation)):
            run = i + 1
            print("Run {}: Column Shifts / Classification Report".format(run))
            explanation = self.explanation[i]['data']
            information = self.information[i]

            if is_in_jupyter():
                relative_accuracy = self.explanation[i]['relative_accuracy']
                explanation = explanation.style.apply(
                    lambda x: ['background: #FF6A6A' if x['accuracy in %'] < relative_accuracy else ''] * len(x),
                    axis=1)
                html_str = ''
                html_str += explanation.render()
                html_str += information.to_html()
                display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)
            else:
                print(explanation, '\n')
                print(information, '\n')

    def print_report(self):
        nprint(self.check_name, text_formatting='h2')
        print("Examined Columns: {}".format(self.examined_columns))
        print("Shifted Columns: {}".format(self.shifted_columns))
        print()

        self.print_explanation()
