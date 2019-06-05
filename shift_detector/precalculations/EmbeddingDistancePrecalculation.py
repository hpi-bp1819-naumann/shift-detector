from shift_detector.precalculations.TextEmbeddingPrecalculation import TextEmbeddingPrecalculation
from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.Precalculation import Precalculation
from datawig.utils import random_split
from math import sqrt


class EmbeddingDistancePrecalculation(Precalculation):

    def __init__(self, model=None, trained_model=None):
        self.model = model
        self.trained_model = trained_model

    def __eq__(self, other):
        return self.model == other.model \
               and self.trained_model == other.trained_model

    def __hash__(self):
        return hash((self.model, self.trained_model))

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

    def process(self, store: Store) -> dict:
        """
        Calculate the Sørensen dice coefficient between two columns
        :param store:
        :return: CheckResult
        """
        # TODO: Catch Error when Dataset is to small (<20 entries) -> split results in empty set

        df1, df2 = store[TextEmbeddingPrecalculation(model=self.model, trained_model=self.trained_model)]

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

        return result
