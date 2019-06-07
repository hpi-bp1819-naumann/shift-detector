from collections import defaultdict
from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.EmbeddingDistancePrecalculation import EmbeddingDistancePrecalculation


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
            explanation[column_name] = "\tBaseline in Dataset1: {}\n" \
                                       "\tBaseline in Dataset2: {}\n" \
                                       "\tDistance between Datasets: {}"\
                .format(data[column_name][0], data[column_name][1], data[column_name][2])

            if (abs(data[column_name][0] - data[column_name][1]) > data[column_name][0] * 3
                    or abs(data[column_name][0] - data[column_name][1]) > data[column_name][1] * 3
                    or abs(data[column_name][0] - data[column_name][2]) > data[column_name][0] * 3
                    or abs(data[column_name][1] - data[column_name][2]) > data[column_name][1] * 3):
                shifted_columns.add(column_name)

        return Report("Embedding Distance Check", examined_columns, shifted_columns, dict(explanation))
