from collections import defaultdict

import matplotlib.pyplot as plt

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.ConditionalProbabilitiesPrecalculation import ConditionalProbabilitiesPrecalculation
from shift_detector.precalculations.conditional_probabilities.fpgrowth import get_columns, to_string


class ConditionalProbabilitiesCheck(Check):
    """
    The ConditionalProbabilitiesCheck object implements the conditional probabilities check.
    See also :ref:`conditional_probabilities_parameters`.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        required both in terms of memory and CPU.
    :param min_confidence: a float between 0 and 1. This parameter impacts the size
        of the result. The higher ``min_confidence`` the more rules are generated.
    :param rule_limit: an int greater than or equal to 0. At most ``rule_limit``
        rules are printed as a result of executing this check. The rules are sorted
        according to their significance.
    :param min_delta_supports: a float between 0 and 1. Rules whose support
        values exhibit an absolute difference of less than ``min_delta_supports`` are pruned.
    :param min_delta_confidences: a float between 0 and 1. Rules whose confidence
        values exhibit an absolute difference of less than ``min_delta_confidences`` are pruned.
    """

    def __init__(self, min_support=0.01, min_confidence=0.15, rule_limit=5, min_delta_supports=0.05,
                 min_delta_confidences=0.05):
        assert 0 <= min_support <= 1, 'min_support expects a float between 0 and 1'
        assert 0 <= min_confidence <= 1, 'min_confidence expects a float between 0 and 1'
        assert 0 <= rule_limit, 'rule_limit expects an int greater than or equal to 0'
        assert 0 <= min_delta_supports <= 1, 'min_delta_supports expects a float between 0 and 1'
        assert 0 <= min_delta_confidences <= 1, 'min_delta_confidences expects a float between 0 and 1'
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.rule_limit = int(rule_limit)
        self.min_delta_supports = float(min_delta_supports)
        self.min_delta_confidences = float(min_delta_confidences)

    def run(self, store):
        rules, examined_columns = store[
            ConditionalProbabilitiesPrecalculation(self.min_support, self.min_confidence)]

        shifted_columns = set()
        explanation = defaultdict(list)
        count = 0
        for rule in rules:
            if count == self.rule_limit:
                break

            if abs(rule.delta_supports) >= self.min_delta_supports and abs(
                    rule.delta_confidences) >= self.min_delta_confidences:
                columns = get_columns(rule)
                shifted_columns.add(columns)
                explanation[', '.join(columns)].append(to_string(rule))

                count += 1

        def plot_result():
            x = [abs(rule.delta_supports) for rule in rules]
            y = [abs(rule.delta_confidences) for rule in rules]
            plt.scatter(x, y)
            plt.title('Conditional Probabilities')
            plt.xlabel('Absolute delta supports')
            plt.ylabel('Absolute delta confidences')
            plt.show()

        return Report('Conditional Probabilities', examined_columns, shifted_columns, explanation,
                      figures=[plot_result])
