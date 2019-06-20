from collections import defaultdict

import matplotlib.pyplot as plt

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.conditional_probabilities_precalculation import \
    ConditionalProbabilitiesPrecalculation


class ConditionalProbabilitiesReport(Report):
    def print_explanation(self):
        if 'significant_rules_of_first' in self.explanation or 'significant_rules_of_second' in self.explanation:
            def to_str(rule, index):
                return '\tSupport: {:.0%}\n\t[{}]'.format(rule.supports[index], ', '.join(
                    f'{attr}: {value}' for attr, value in rule.left_side + rule.right_side))

            if 'significant_rules_of_first' in self.explanation:
                print('Values exclusive to first data set:\n')
                print('\n\n'.join(to_str(rule, 0) for rule in self.explanation['significant_rules_of_first']),
                      end='\n\n')
            if 'significant_rules_of_second' in self.explanation:
                print('Values exclusive to second data set:\n')
                print('\n\n'.join(to_str(rule, 1) for rule in self.explanation['significant_rules_of_second']),
                      end='\n\n')
        if 'mutual_rules' in self.explanation:
            print('Mutual rules:\n')
            for rule in self.explanation['mutual_rules']:
                print('\t{}\n'.format(rule))


class ConditionalProbabilitiesCheck(Check):
    """
    The ConditionalProbabilitiesCheck object implements the :ref:`conditional_probabilities` check.
    For more information about its parameters see also :ref:`conditional_probabilities_parameters`.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        required during computation both in terms of memory and CPU.
    :param min_confidence: a float between 0 and 1. This parameter impacts the size
        of the result. The higher ``min_confidence`` the more rules are considered.
    :param rule_limit: an int greater than or equal to 0. At most ``rule_limit``
        rule clusters are printed as a result of executing this check.
        The rule clusters are sorted according to their significance.
    :param min_delta_supports: a float between 0 and 1. Rules whose support
        values exhibit an absolute difference of less than ``min_delta_supports`` are not considered.
    :param min_delta_confidences: a float between 0 and 1. Rules whose confidence
        values exhibit an absolute difference of less than ``min_delta_confidences`` are not considered.
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
        pre_calculation = store[
            ConditionalProbabilitiesPrecalculation(self.min_support, self.min_confidence,
                                                   self.number_of_bins, self.number_of_topics,
                                                   self.min_delta_supports, self.min_delta_confidences)
        ]

        shifted_columns = set()
        explanation = defaultdict(list)

        def add_to_explanation(rules, name):
            for i, rule in enumerate(rules):
                if i == self.rule_limit:
                    break
                columns = tuple(sorted(attribute for attribute, _ in rule.left_side + rule.right_side))
                shifted_columns.add(columns)
                explanation[name].append(rule)

        add_to_explanation(pre_calculation.significant_rules_of_first, 'significant_rules_of_first')
        add_to_explanation(pre_calculation.significant_rules_of_second, 'significant_rules_of_second')
        add_to_explanation(pre_calculation.sorted_filtered_mutual_rules, 'mutual_rules')

        def plot_result():
            coordinates = [(abs(rule.delta_supports), abs(rule.delta_confidences)) for rule in
                           pre_calculation.mutual_rules]
            green_x = [x for x, y in coordinates if x <= self.min_delta_supports and y <= self.min_delta_confidences]
            green_y = [y for x, y in coordinates if x <= self.min_delta_supports and y <= self.min_delta_confidences]
            orange_x = [x for x, y in coordinates if x > self.min_delta_supports
                        and y <= self.min_delta_confidences or x <= self.min_delta_supports
                        and y > self.min_delta_confidences]
            orange_y = [y for x, y in coordinates if x > self.min_delta_supports
                        and y <= self.min_delta_confidences or x <= self.min_delta_supports
                        and y > self.min_delta_confidences]
            red_x = [x for x, y in coordinates if x > self.min_delta_supports and y > self.min_delta_confidences]
            red_y = [y for x, y in coordinates if x > self.min_delta_supports and y > self.min_delta_confidences]
            plt.scatter(green_x, green_y, color='green')
            plt.scatter(orange_x, orange_y, color='orange')
            plt.scatter(red_x, red_y, color='red')
            plt.title('Mutual conditional probabilities')
            plt.xlabel('Absolute delta supports')
            plt.ylabel('Absolute delta confidences')
            plt.xticks([i / 10 for i in range(0, 11)])
            plt.yticks([i / 10 for i in range(0, 11)])
            plt.axhline(y=self.min_delta_confidences, linestyle='--', linewidth=2, color='black')
            plt.axvline(x=self.min_delta_supports, linestyle='--', linewidth=2, color='black')
            plt.show()

        return ConditionalProbabilitiesReport('Conditional Probabilities', shifted_columns, shifted_columns,
                                              explanation, figures=[plot_result])
