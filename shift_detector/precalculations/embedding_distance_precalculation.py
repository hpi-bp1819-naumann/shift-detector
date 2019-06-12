from shift_detector.precalculations.text_embedding_precalculation import TextEmbeddingPrecalculation
from shift_detector.precalculations.store import Store
from shift_detector.precalculations.precalculation import Precalculation
from datawig.utils import random_split
import numpy as np
from numpy.linalg import norm


class EmbeddingDistancePrecalculation(Precalculation):

    def __init__(self, model=None, trained_model=None):
        self.model = model
        self.trained_model = trained_model

    def __eq__(self, other):
        return self.model == other.model and self.trained_model == other.trained_model

    def __hash__(self):
        return hash((self.model, self.trained_model))

    @staticmethod
    def sum_and_normalize_vectors(series):
        vector = np.array([0.0] * len(series.iloc[0]))
        for cell in series:
            vector += cell
        return vector / len(series)

    def process(self, store: Store) -> dict:
        """
        Calculate the euclidean distance between two embeddings.
        :param store:
        :return: CheckResult
        """

        df1, df2 = store[TextEmbeddingPrecalculation(model=self.model, trained_model=self.trained_model, agg='sum')]

        df1a, df1b = random_split(df1, [0.95, 0.05])           # Baseline for df1
        df2a, df2b = random_split(df2, [0.95, 0.05])           # Baseline for df2

        if df1a.empty or df1b.empty or df2a.empty or df2b.empty:
            raise ValueError('Dataset to small for split ratio')

        result = {}
        for i in df1:
            result[i] = (norm(self.sum_and_normalize_vectors(df1a[i]) - self.sum_and_normalize_vectors(df1b[i])),
                         norm(self.sum_and_normalize_vectors(df2a[i]) - self.sum_and_normalize_vectors(df2b[i])),
                         norm(self.sum_and_normalize_vectors(df1[i]) - self.sum_and_normalize_vectors(df2[i])))

        return result
