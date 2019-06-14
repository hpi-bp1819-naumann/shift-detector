import pandas as pd

from shift_detector.precalculations.binning_precalculation import BinningPrecalculation
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.precalculations.conditional_probabilities import fpgrowth
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType


class ConditionalProbabilitiesPrecalculation(Precalculation):

    def __init__(self, min_support, min_confidence, number_of_bins, number_of_topics):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.number_of_bins = number_of_bins
        self.number_of_topics = number_of_topics

    def process(self, store):
        df1_cat, df2_cat = store[ColumnType.categorical]
        df1_num, df2_num = store[BinningPrecalculation(self.number_of_bins)]
        df1_text, df2_text = store[LdaEmbedding(self.number_of_topics)]

        df1 = pd.concat([df1_cat, df1_num, df1_text], axis=1)
        df2 = pd.concat([df2_cat, df2_num, df2_text], axis=1)

        rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence)
        return rules, list(df1.columns)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics) == (
            other.min_support, other.min_confidence, other.number_of_bins, other.number_of_topics)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics))
