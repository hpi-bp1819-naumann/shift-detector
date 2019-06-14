from collections import defaultdict

import matplotlib.pyplot as plt

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.conditional_probabilities import rule_compression
from shift_detector.precalculations.conditional_probabilities_precalculation import \
    ConditionalProbabilitiesPrecalculation


class ConditionalProbabilitiesReport(Report):
    def explanation_str(self):
        return '\n\n'.join('{}'.format('\n'.join(rules)) for rules in self.explanation.values())


class ConditionalProbabilitiesCheck(Check):
    """
    The ConditionalProbabilitiesCheck object implements the :ref:`conditional_probabilities` check.
    For more information about its parameters see also :ref:`conditional_probabilities_parameters`.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        required both in terms of memory and CPU.
    :param min_confidence: a float between 0 and 1. This parameter impacts the size
        of the result. The higher ``min_confidence`` the more rules are generated.
    :param rule_limit: an int greater than or equal to 0. At most ``rule_limit``
        rules are printed as a result of executing this check. The rules are sorted
        according to their significance.
    :param min_delta_supports: a float between 0 and 1. Rules whose support
        values exhibit an absolute difference of less than ``min_delta_supports`` are not printed.
    :param min_delta_confidences: a float between 0 and 1. Rules whose confidence
        values exhibit an absolute difference of less than ``min_delta_confidences`` are not printed.
    :param number_of_bins: an int greater than 0. Numerical columns are binned into ``number_of_bins``
        bins.
    :param number_of_topics: an int greater than 0. Textual columns are embedded into ``number_of_topics``
        topics.
    """

    def __init__(self, min_support=0.01, min_confidence=0.15, rule_limit=5, min_delta_supports=0.05,
                 min_delta_confidences=0.05, number_of_bins=50, number_of_topics=20):
        assert 0 <= min_support <= 1, 'min_support expects a float between 0 and 1'
        assert 0 <= min_confidence <= 1, 'min_confidence expects a float between 0 and 1'
        assert 0 <= rule_limit, 'rule_limit expects an int greater than or equal to 0'
        assert 0 <= min_delta_supports <= 1, 'min_delta_supports expects a float between 0 and 1'
        assert 0 <= min_delta_confidences <= 1, 'min_delta_confidences expects a float between 0 and 1'
        assert 0 < number_of_bins, 'number_of_bins expects an int greater than 0'
        assert 0 < number_of_topics, 'number_of_topics expects an int greater than 0'
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.rule_limit = int(rule_limit)
        self.min_delta_supports = float(min_delta_supports)
        self.min_delta_confidences = float(min_delta_confidences)
        self.number_of_bins = int(number_of_bins)
        self.number_of_topics = int(number_of_topics)

    def run(self, store):
        rules, examined_columns = store[
            ConditionalProbabilitiesPrecalculation(self.min_support, self.min_confidence, self.number_of_bins,
                                                   self.number_of_topics)]

        filtered_rules = (rule for rule in rules if abs(rule.delta_supports) >= self.min_delta_supports and abs(
            rule.delta_confidences) >= self.min_delta_confidences)

        compressed_rules = rule_compression.compress_and_sort_rules(filtered_rules)

        shifted_columns = set()
        explanation = defaultdict(list)
        for i, compressed_rule in enumerate(compressed_rules):
            if i == self.rule_limit:
                break
            columns = tuple(sorted(attribute for attribute, _ in compressed_rule.attribute_value_pairs))
            shifted_columns.add(columns)

            explanation[', '.join(columns)].append(str(compressed_rule))

        def plot_result():
            x = [abs(rule.delta_supports) for rule in rules]
            y = [abs(rule.delta_confidences) for rule in rules]
            plt.scatter(x, y)
            plt.title('Conditional Probabilities')
            plt.xlabel('Absolute delta supports')
            plt.ylabel('Absolute delta confidences')
            plt.xticks([i / 10 for i in range(0, 11)])
            plt.yticks([i / 10 for i in range(0, 11)])
            plt.show()

        return ConditionalProbabilitiesReport('Conditional Probabilities', examined_columns, shifted_columns,
                                              explanation, figures=[plot_result])
