from shift_detector.Utils import ColumnType
from shift_detector.precalculations.frequent_item_rules import fpgrowth
from shift_detector.precalculations.frequent_item_rules import rule_compression
from shift_detector.precalculations.Precalculation import Precalculation


class FrequentItemsetPrecalculation(Precalculation):

    def __init__(self, min_support=0.01, min_confidence=0.15):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def process(self, store):
        df1, df2 = store[ColumnType.categorical]

        item_rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence)
        compressed_rules = rule_compression.compress_rules(item_rules)

        return {'compressed_rules': compressed_rules, 'examined_columns': set(df1.columns)}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.min_support == other.min_support and self.min_confidence == other.min_confidence

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence))
