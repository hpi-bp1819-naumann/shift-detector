from collections import defaultdict

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.ConditionalProbabilitiesPrecalculation import ConditionalProbabilitiesPrecalculation


class ConditionalProbabilitiesCheck(Check):
    """
    The ConditionalProbabilitiesCheck object implements the conditional probabilities check.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        required during computation both in terms of memory and CPU. The default
        value is 0.01, which is high enough to get a reasonably good performance
        and still low enough to not prematurely exclude significant association rules.
        This parameter allows you to adjust the granularity of the comparison of the
        two data sets.
    :param min_confidence: a float between 0 and 1. This parameter impacts the amount
        of generated association rules. The higher ``min_confidence`` the more rules
        are generated.
    :param rule_limit: this parameter expects an int. At most
        ``rule_limit`` rules are printed as a result of executing this check.
        The rules are sorted according to their significance.
    :param min_delta_supports: a float between 0 and 1. Association rules whose support
        values exhibit a difference of less than ``min_delta_supports`` are pruned.
    :param min_delta_confidences: a float between 0 and 1. Association rules whose confidence
        values exhibit a difference of less than ``min_delta_confidences`` are pruned.
    """

    def __init__(self, min_support=0.01, min_confidence=0.15, rule_limit=5, min_delta_supports=0.05,
                 min_delta_confidences=0.05):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rule_limit = rule_limit
        self.min_delta_supports = min_delta_supports
        self.min_delta_confidences = min_delta_confidences

    def run(self, store):
        """Compute conditional probabilities, compress them and return a report."""
        compressed_rules, examined_columns = store[
            ConditionalProbabilitiesPrecalculation(self.min_support, self.min_confidence, self.min_delta_supports,
                                                   self.min_delta_confidences)]

        shifted_columns = set()
        explanation = defaultdict(list)
        for i, compressed_rule in enumerate(compressed_rules):
            if i == self.rule_limit:
                break
            columns = tuple(sorted(key for key, _ in compressed_rule.attributes))
            shifted_columns.add(columns)

            explanation[', '.join(columns)].append(str(compressed_rule))

        return Report(examined_columns, list(shifted_columns), explanation)
