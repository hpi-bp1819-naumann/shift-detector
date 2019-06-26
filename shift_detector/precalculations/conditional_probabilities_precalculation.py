from collections import namedtuple
from itertools import combinations

import pandas as pd

from shift_detector.precalculations.binning_precalculation import BinningPrecalculation
from shift_detector.precalculations.conditional_probabilities import fpgrowth
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType


class ConditionalProbabilitiesPrecalculation(Precalculation):
    def __init__(self, min_support, min_confidence, number_of_bins, number_of_topics,
                 min_delta_supports, min_delta_confidences):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.number_of_bins = number_of_bins
        self.number_of_topics = number_of_topics
        self.min_delta_supports = min_delta_supports
        self.min_delta_confidences = min_delta_confidences

    def process(self, store):
        df1_cat, df2_cat = store[ColumnType.categorical]
        df1_num, df2_num, _ = store[BinningPrecalculation(self.number_of_bins)]

        df1 = pd.concat([df1_cat, df1_num], axis=1)
        df2 = pd.concat([df2_cat, df2_num], axis=1)

        rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support, self.min_confidence)

        exclusive_rules_of_first = []  # support of left side is 0 in second data set
        exclusive_rules_of_second = []  # support of left side is 0 in first data set
        mutual_rules = []  # support of left side is not 0 in both data sets and right side is not empty

        for rule in rules:
            if not rule.supports_of_left_side[1]:
                exclusive_rules_of_first.append(rule)
            elif not rule.supports_of_left_side[0]:
                exclusive_rules_of_second.append(rule)
            elif rule.right_side:  # rules without a right side do not add any further information
                mutual_rules.append(rule)

        filtered_mutual_rules = (rule for rule in mutual_rules if
                                 abs(rule.delta_supports) >= self.min_delta_supports and abs(
                                     rule.delta_confidences) >= self.min_delta_confidences)
        sorted_filtered_mutual_rules = sorted(filtered_mutual_rules,
                                              key=lambda r: (
                                                  min(r.supports) != 0, abs(r.delta_confidences),
                                                  max(r.supports_of_left_side)),
                                              reverse=True)

        def get_significant_rules(rules, index):
            significant_rules = []
            smallest_sub_sets = set()

            for rule in sorted(rules, key=lambda r: (len(r.left_side), len(r.right_side))):
                for length in range(1, len(rule.left_side) + 1):
                    if any(True for sub_set in combinations(rule.left_side, length) if sub_set in smallest_sub_sets):
                        break
                else:
                    smallest_sub_sets.add(rule.left_side)
                    significant_rules.append(rule)

            return sorted(significant_rules, key=lambda r: (r.supports_of_left_side[index], r.supports[index]),
                          reverse=True)

        Result = namedtuple('Result', ['examined_columns', 'significant_rules_of_first', 'significant_rules_of_second',
                                       'sorted_filtered_mutual_rules', 'mutual_rules', 'total_number_of_rules'])
        return Result(examined_columns=list(df1.columns),
                      significant_rules_of_first=get_significant_rules(exclusive_rules_of_first, 0),
                      significant_rules_of_second=get_significant_rules(exclusive_rules_of_second, 1),
                      sorted_filtered_mutual_rules=sorted_filtered_mutual_rules,
                      mutual_rules=mutual_rules,
                      total_number_of_rules=len(rules)
                      )

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
