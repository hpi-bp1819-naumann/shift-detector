from collections import defaultdict
from morpheus.checks.check import Check, Report
from morpheus.precalculations.embedding_distance_precalculation import EmbeddingDistancePrecalculation
from IPython.display import display

import pandas as pd


class EmbeddingDistanceCheck(Check):

    def __init__(self, model='word2vec', trained_model=None):
        self.model = model
        self.trained_model = trained_model

    def run(self, store):
        data = store[EmbeddingDistancePrecalculation(model=self.model, trained_model=self.trained_model)]

        examined_columns = set(data.keys())
        shifted_columns = set()
        explanation = defaultdict(str)

        for column_name in data:
            baseline1, baseline2, distance = data[column_name]

            if (abs(baseline1 - baseline2) > baseline1 * 3
                    or abs(baseline1 - baseline2) > baseline2 * 3
                    or abs(baseline1 - distance) > baseline1 * 3
                    or abs(baseline2 - distance) > baseline2 * 3):
                shifted_columns.add(column_name)

        return EmbeddingDistanceReport("Embedding Distance Check", examined_columns, shifted_columns, dict(explanation), information=data)


class EmbeddingDistanceReport(Report):

    def print_information(self):
        result_df = pd.DataFrame.from_dict(self.information, orient='index')
        result_df.columns = ['Baseline in Dataset 1', 'Baseline in Dataset 2', 'Distance between Datasets']
        display(result_df)
