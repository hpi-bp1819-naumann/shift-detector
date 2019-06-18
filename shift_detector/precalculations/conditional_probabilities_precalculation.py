import pandas as pd

from shift_detector.precalculations.binning_precalculation import BinningPrecalculation
from shift_detector.precalculations.conditional_probabilities import fpgrowth
from shift_detector.precalculations.conditional_probabilities import rule_compression
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
        # df1_text, df2_text = store[LdaEmbedding(self.number_of_topics)]

        # df1 = pd.concat([df1_cat, df1_num, df1_text], axis=1)
        # df2 = pd.concat([df2_cat, df2_num, df2_text], axis=1)
        df1 = pd.concat([df1_cat, df1_num], axis=1)
        df2 = pd.concat([df2_cat, df2_num], axis=1)

        rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence)
        return rules, list(df1.columns)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics) == (
            other.min_support, other.min_confidence, other.number_of_bins, other.number_of_topics)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics))


class ConditionalProbabilitiesCompressionPrecalculation(Precalculation):
    def __init__(self, min_support, min_confidence, number_of_bins, number_of_topics,
                 min_delta_supports, min_delta_confidences, rules):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.number_of_bins = number_of_bins
        self.number_of_topics = number_of_topics
        self.min_delta_supports = min_delta_supports
        self.min_delta_confidences = min_delta_confidences
        self.rules = rules

    def process(self, store):
        filtered_rules = (rule for rule in self.rules if abs(rule.delta_supports) >= self.min_delta_supports and abs(
            rule.delta_confidences) >= self.min_delta_confidences)

        return rule_compression.compress_and_sort_rules(filtered_rules)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics,
                self.min_delta_supports, self.min_delta_confidences) == (
                   other.min_support, other.min_confidence, other.number_of_bins, other.number_of_topics,
                   other.min_delta_supports, other.min_delta_confidences)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence, self.number_of_bins, self.number_of_topics,
                     self.min_delta_supports, self.min_delta_confidences))
