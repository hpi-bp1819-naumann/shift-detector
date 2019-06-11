import pandas as pd

from shift_detector.precalculations.BinningPrecalculation import BinningPrecalculation
from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.precalculations.conditional_probabilities import fpgrowth
from shift_detector.utils.ColumnManagement import ColumnType
from shift_detector.precalculations.conditional_probabilities import rule_compression


class ConditionalProbabilitiesPrecalculation(Precalculation):

    def __init__(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def process(self, store):
        df1_cat, df2_cat = store[ColumnType.categorical]
        df1_num, df2_num = store[BinningPrecalculation()]

        df1 = pd.concat([df1_cat, df1_num], axis=1)
        df2 = pd.concat([df2_cat, df2_num], axis=1)

        uncompressed_rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence)
        compressed_rules = rule_compression.compress_rules(uncompressed_rules)
        return compressed_rules, list(df1.columns)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence) == (other.min_support, other.min_confidence)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence))
