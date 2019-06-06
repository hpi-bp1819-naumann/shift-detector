from shift_detector.Utils import ColumnType
from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.precalculations.conditional_probabilities import fpgrowth


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
        return rules, set(df1.columns)  # TODO: Why is this a set?

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.min_support, self.min_confidence, self.min_delta_supports, self.min_delta_confidences) == (
            other.min_support, other.min_confidence, other.min_delta_supports, other.min_delta_confidences)

    def __hash__(self):
        return hash((self.__class__, self.min_support, self.min_confidence, self.min_delta_supports,
                     self.min_delta_confidences))
