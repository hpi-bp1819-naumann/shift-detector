from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.embedding_distance_precalculation import EmbeddingDistancePrecalculation
from IPython.display import display

import pandas as pd


class EmbeddingDistanceCheck(Check):

    def __init__(self, model='word2vec', trained_model=None, threshold=3.0):
        self.model = model
        self.trained_model = trained_model
        self.threshold = threshold

    def run(self, store):
        data = store[EmbeddingDistancePrecalculation(model=self.model, trained_model=self.trained_model)]

        examined_columns = set(data.keys())
        shifted_columns = set()

        for column_name in data:
            baseline1, baseline2, distance = data[column_name]

            if (abs(baseline1 - baseline2) > baseline1 * self.threshold
                    or abs(baseline1 - baseline2) > baseline2 * self.threshold
                    or abs(baseline1 - distance) > baseline1 * self.threshold
                    or abs(baseline2 - distance) > baseline2 * self.threshold):
                shifted_columns.add(column_name)

        return EmbeddingDistanceReport("Embedding Distance Check", examined_columns, shifted_columns,
                                       information=(self.threshold, data))


class EmbeddingDistanceReport(Report):

    def print_information(self):
        result_df = pd.DataFrame.from_dict(self.information[1], orient='index')
        result_df.columns = ['Distance within Dataset 1', 'Distance within Dataset 2',
                             'Distance between Datasets']
        result_df['Rel. Threshold'] = self.information[0]
        display(result_df.loc[self.shifted_columns])
