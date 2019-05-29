from shift_detector.preprocessors.TextEmbedding import TextEmbedding, EmbeddingType
from shift_detector.preprocessors.Store import Store
from shift_detector.checks.Check import Check, Report
from datawig.utils import random_split
from math import sqrt


class SorensenDiceReport(Report):
    def __init__(self, result: dict):
        self.result = result

    def print_report(self):
        print('Sorensen Dice Report')
        for i in self.result:
            print(i, self.result[i])


class EmbeddingDistanceCheck(Check):

    @staticmethod
    def calculate_distance(vector1, vector2):
        delta_vector = [0] * len(vector1)
        for idx, val in enumerate(vector1):
            delta_vector[idx] = vector1[idx] - vector2[idx]
        distance = 0
        for i in delta_vector:
            distance += i ** 2
        return sqrt(distance)

    @staticmethod
    def join_and_normalize_vectors(series):
        vector = [0.0] * len(series.iloc[0])
        for cell in series:
            for idx, val in enumerate(cell):
                vector[idx] += float(val)
        count = len(series)
        for idx, val in enumerate(vector):
            vector[idx] = val / count
        return vector

    def run(self, store: Store) -> SorensenDiceReport:
        """
        Calculate the Sørensen dice coefficient between two columns
        :param store:
        :return: CheckResult
        """

        df1, df2 = store[TextEmbedding(model=EmbeddingType.Word2Vec)]

        df1a, df1b = random_split(df1, [0.95, 0.05], seed=11)           # Baseline for df1
        df2a, df2b = random_split(df2, [0.95, 0.05], seed=11)           # Baseline for df2

        result = {}
        for i in df1:

            result[i] = (self.calculate_distance(self.join_and_normalize_vectors(df1a[i]),
                                                 self.join_and_normalize_vectors(df1b[i])),
                         self.calculate_distance(self.join_and_normalize_vectors(df2a[i]),
                                                 self.join_and_normalize_vectors(df2b[i])),
                         self.calculate_distance(self.join_and_normalize_vectors(df1[i]),
                                                 self.join_and_normalize_vectors(df2[i])))

        return SorensenDiceReport(result)