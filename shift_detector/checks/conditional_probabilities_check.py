from collections import defaultdict

import matplotlib.pyplot as plt

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.conditional_probabilities_precalculation import \
    ConditionalProbabilitiesPrecalculation
from shift_detector.utils.custom_print import nprint, mdprint


class ConditionalProbabilitiesReport(Report):
    def __init__(self, check_name, examined_columns, shifted_column_combinations, explanation={}, information={},
                 figures=[]):
        super().__init__(check_name, examined_columns=[], shifted_columns=[], explanation=explanation,
                         information=information, figures=figures)
        self.shifted_column_combinations = shifted_column_combinations
        self.__examined_columns = examined_columns

    def print_report(self):
        nprint(self.check_name, text_formatting='h2')
        print("Considered Columns: {}".format(self.__examined_columns))
        print("Shifted Column Combinations: {}\n".format(self.shifted_column_combinations))
        self.print_explanation()
        self.print_information()

    def print_explanation(self):
        if 'significant_rules_of_first' in self.explanation or 'significant_rules_of_second' in self.explanation:
            def to_str(rule, index):
                return 'Support: {:.0%}\n[{}]\n'.format(rule.supports[index], ', '.join(
                    f'{attr}: {value}' for attr, value in rule.left_side + rule.right_side))

            if 'significant_rules_of_first' in self.explanation:
                nprint('Attribute-value combinations exclusive to first data set', text_formatting='h3')
                print('\n'.join(to_str(rule, 0) for rule in self.explanation['significant_rules_of_first']))
            if 'significant_rules_of_second' in self.explanation:
                nprint('Attribute-value combinations exclusive to second data set', text_formatting='h3')
                print('\n'.join(to_str(rule, 1) for rule in self.explanation['significant_rules_of_second']))
        if 'sorted_filtered_mutual_rules' in self.explanation:
            nprint('Red rules', text_formatting='h3')
            mdprint('**Red rules exceed both min_delta_supports and min_delta_confidences.**\n')
            mdprint('The first rule can be read as follows:\n')
            for i, rule in enumerate(self.explanation['sorted_filtered_mutual_rules']):
                if i == 0:
                    explanation_string = '*If the condition holds that '
                    explanation_string += ' and '.join('{}={}'.format(attr, value) for attr, value in rule.left_side)
                    explanation_string += ' then the probability that '
                    explanation_string += ' and '.join('{}={}'.format(attr, value) for attr, value in rule.right_side)
                    explanation_string += ' is {:.0%} in the first data set and {:.0%} in the second data set.*\n'.format(
                        rule.confidences[0], rule.confidences[1])
                    explanation_string += '*The attribute-value combination '
                    explanation_string += ' and '.join('{}={}'.format(attr, value) for attr, value in rule.left_side)
                    explanation_string += ' appears in {:.0%} of the tuples in the first data set and in {:.0%} of the tuples in the second data set.*\n'.format(
                        rule.supports_of_left_side[0], rule.supports_of_left_side[1])
                    explanation_string += '*The attribute-value combination '
                    explanation_string += ' and '.join(
                        '{}={}'.format(attr, value) for attr, value in rule.left_side + rule.right_side)
                    explanation_string += ' appears in {:.0%} of the tuples in the first data set and in {:.0%} of the tuples in the second data set.*\n'.format(
                        rule.supports[0], rule.supports[1])
                    mdprint(explanation_string)
                print(rule, end='\n\n')
                if min(rule.supports) == 0 and i > 0 and min(
                        self.explanation['sorted_filtered_mutual_rules'][i - 1].supports) != 0:
                    # insert visual marker
                    print('\n')
        if ('orange_rules_falling_below_min_delta_supports' in self.explanation
                or 'orange_rules_falling_below_min_delta_confidences' in self.explanation):
            nprint('Orange rules', text_formatting='h3')
            if 'orange_rules_falling_below_min_delta_supports' in self.explanation:
                mdprint('**The following rules fall below min_delta_supports but exceed min_delta_confidences:**\n')
                print('\n\n'.join(
                    str(rule) for rule in self.explanation['orange_rules_falling_below_min_delta_supports']),
                    end='\n\n')
            if 'orange_rules_falling_below_min_delta_confidences' in self.explanation:
                mdprint('**The following rules fall below min_delta_confidences but exceed min_delta_supports:**\n')
                print('\n\n'.join(
                    str(rule) for rule in self.explanation['orange_rules_falling_below_min_delta_confidences']),
                    end='\n\n')


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
        rules are printed in each section of the report as a result of executing this check.
        Rules are always sorted according to their significance.
    :param min_delta_supports: a float between 0 and 1. Rules whose support
        values exhibit an absolute difference of less than ``min_delta_supports`` are not considered
        to indicate a shift.
    :param min_delta_confidences: a float between 0 and 1. Rules whose confidence
        values exhibit an absolute difference of less than ``min_delta_confidences`` are not considered
        to indicate a shift.
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
        pre_calculation_result = store[
            ConditionalProbabilitiesPrecalculation(self.min_support, self.min_confidence,
                                                   self.number_of_bins, self.number_of_topics,
                                                   self.min_delta_supports, self.min_delta_confidences)
        ]
        explanation = defaultdict(list)

        def add_to_explanation(rules, identifier):
            for i, rule in enumerate(rules):
                if i == self.rule_limit:
                    break
                explanation[identifier].append(rule)

        add_to_explanation(pre_calculation_result.significant_rules_of_first, 'significant_rules_of_first')
        add_to_explanation(pre_calculation_result.significant_rules_of_second, 'significant_rules_of_second')
        add_to_explanation(pre_calculation_result.sorted_filtered_mutual_rules, 'sorted_filtered_mutual_rules')

        orange_rules_falling_below_min_delta_supports = sorted(
            (rule for rule in pre_calculation_result.mutual_rules if abs(rule.delta_supports) < self.min_delta_supports
             and abs(rule.delta_confidences) >= self.min_delta_confidences),
            key=lambda r: (min(r.supports) != 0, abs(r.delta_confidences) * abs(r.delta_supports)),
            reverse=True
        )
        add_to_explanation(orange_rules_falling_below_min_delta_supports,
                           'orange_rules_falling_below_min_delta_supports')
        orange_rules_falling_below_min_delta_confidences = sorted(
            (rule for rule in pre_calculation_result.mutual_rules if
             abs(rule.delta_confidences) < self.min_delta_confidences
             and abs(rule.delta_supports) >= self.min_delta_supports),
            key=lambda r: (min(r.supports) != 0, abs(r.delta_confidences) * abs(r.delta_supports)),
            reverse=True
        )
        add_to_explanation(orange_rules_falling_below_min_delta_confidences,
                           'orange_rules_falling_below_min_delta_confidences')

        shifted_columns = set(tuple(sorted(attribute for attribute, _ in rule.left_side + rule.right_side))
                              for rule in pre_calculation_result.sorted_filtered_mutual_rules)

        def plot_result():
            coordinates = [(abs(rule.delta_supports), abs(rule.delta_confidences)) for rule in
                           pre_calculation_result.mutual_rules]
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
            plt.rcParams['figure.figsize'] = (10, 8)
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

        return ConditionalProbabilitiesReport('Conditional Probabilities', pre_calculation_result.examined_columns,
                                              shifted_columns, explanation, figures=[plot_result])
