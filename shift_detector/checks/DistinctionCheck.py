from sklearn.metrics import classification_report

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.DistinctionPrecalculation import DistinctionPrecalculation


class DistinctionCheck(Check):

    def __init__(self, columns=[], num_epochs=10):
        self.columns = columns
        self.num_epochs = num_epochs

    def run(self, store) -> Report:
        result = store[DistinctionPrecalculation(self.columns, self.num_epochs)]
        examined_columns = self.columns
        shifted_columns = self.shifted_columns(result)
        information = self.information(result)

        return Report(examined_columns, shifted_columns, information=information)

    @staticmethod
    def shifted_columns(result):
        return result['relevant_columns']

    @staticmethod
    def information(result):
        y_true = result['y_true']
        y_pred = result['y_pred']

        report = classification_report(y_true, y_pred)
        classification = classification_report(y_true, y_pred, output_dict=True)

        information = dict()
        information['Classification Report'] = report
        information['F1 score df1'] = classification["A"]["f1-score"]
        information['F1 score df2'] = classification["B"]["f1-score"]
        return information
