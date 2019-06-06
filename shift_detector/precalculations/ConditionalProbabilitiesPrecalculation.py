from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.precalculations.conditional_probabilities import fpgrowth
from shift_detector.utils.ColumnManagement import ColumnType


class ConditionalProbabilitiesPrecalculation(Precalculation):

    def __init__(self, min_support, min_confidence, min_delta_supports, min_delta_confidences):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_delta_supports = min_delta_supports
        self.min_delta_confidences = min_delta_confidences

    def process(self, store):
        df1, df2 = store[ColumnType.categorical]
        rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence,
                                                  self.min_delta_supports, self.min_delta_confidences)
        return rules, list(df1.columns)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence, self.min_delta_supports, self.min_delta_confidences) == (
            other.min_support, other.min_confidence, other.min_delta_supports, other.min_delta_confidences)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence, self.min_delta_supports,
                     self.min_delta_confidences))
