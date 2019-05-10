import unittest
from shift_detector.checks.frequent_item_rules import rule_compression
from shift_detector.checks.frequent_item_rules.ExtendendRule import ExtendedRule, RuleCluster
from collections import namedtuple
import copy


class TestFrequentItemRule(unittest.TestCase):

    def setUp(self):
        Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                                   'supports', 'delta_supports', 'confidences', 'delta_confidences'])

        self.single_rule = Rule(left_side=(('value', 'A'),), right_side=(), supports_of_left_side=(0.65, 0.81),
                                delta_supports_of_left_side=-0.16, supports=(0.65, 0.81),
                                delta_supports=-0.16, confidences=(1.0, 1.0), delta_confidences=0.1)
        self.extended_rule = rule_compression.add_side_attributes_to_rules([self.single_rule])[0]

        self.rule_a = copy.copy(self.extended_rule)
        self.rule_b = copy.copy(self.extended_rule)
        self.rule_c = copy.copy(self.extended_rule)

    def test_true(self):
        self.assertTrue(True)

    def test_side_attributes(self):
        self.assertEqual(rule_compression.add_side_attributes_to_rules([]), [])
        self.assertTrue(isinstance(self.extended_rule, ExtendedRule), msg='{0}'.format(self.extended_rule))

        multiple_rules = [self.single_rule, self.single_rule, self.single_rule]
        multiple_rules_extended = rule_compression.add_side_attributes_to_rules(multiple_rules)
        self.assertEqual(len(multiple_rules), len(multiple_rules_extended))

    def test_remove_duplicates(self):
        self.rule_a.left_side = (('value', 'A'), ('value', 'B'), ('value', 'C'))
        self.rule_b.left_side = (('value', 'D'), ('value', 'B'), ('value', 'F'))
        self.rule_c.left_side = (('value', 'X'), ('value', 'B'), ('value', 'Z'))

        duplicates_list = [self.rule_a, self.rule_b, self.rule_c]
        duplicates_list = rule_compression.remove_allsame_attributes(duplicates_list)

        self.assertEqual(len(duplicates_list), 3)
        for rule in duplicates_list:
            self.assertEqual(len(rule.left_side), 2)
            self.assertNotIn(('value', 'B'), rule.left_side)

    def test_true_if_no_attribute_none(self):
        rule_a = copy.copy(self.extended_rule)
        rule_b = copy.copy(self.extended_rule)
        rule_b.left_side = (('value', None),)

        self.assertEqual(rule_compression.true_if_no_attribute_none(rule_a), True)
        self.assertEqual(rule_compression.true_if_no_attribute_none(rule_b), False)

    def test_filter_non_values(self):
        self.rule_a.left_side = (('value', 'A'), ('value', None), ('value', 'C'))
        self.rule_b.left_side = (('value', 'D'), ('value', 'B'), ('value', 'F'))
        self.rule_c.right_side = (('value', 'X'), ('value', 'B'), ('value', None))

        rules = [self.rule_a, self.rule_b, self.rule_c]
        rules = rule_compression.filter_non_values(rules)
        self.assertEqual(len(rules), 1)

    def test_group_by_length(self):
        rule_d = copy.copy(self.extended_rule)
        rule_e = copy.copy(self.extended_rule)
        rule_f = copy.copy(self.extended_rule)
        rule_g = copy.copy(self.extended_rule)

        # 1x1, 3x2, 2x3, 1x4
        self.rule_a.all_sides = (('value', 'A'),)
        rule_e.all_sides = (('value', 'X'), ('value', 'B'))
        rule_f.all_sides = (('value', 'X'), ('value', 'B'))
        self.rule_c.all_sides = (('value', 'X'), ('value', 'B'))
        rule_d.all_sides = (('value', 'X'), ('value', 'U'), ('value', 'I'))
        self.rule_b.all_sides = (('value', 'D'), ('value', 'B'), ('value', 'F'))
        rule_g.all_sides = (('value', 'X'), ('value', 'B'), ('value', 'U'), ('value', 'Y'))
        rules = [self.rule_a, self.rule_b, self.rule_c, rule_d, rule_e, rule_f, rule_g]
        length_groups = rule_compression.group_rules_by_length(rules)

        self.assertEqual(len(length_groups), 4)
        self.assertEqual(len(length_groups[0]), 1)
        self.assertEqual(len(length_groups[1]), 3)
        self.assertEqual(len(length_groups[2]), 2)
        self.assertEqual(len(length_groups[3]), 1)

    def test_cluster_from_rule(self):
        cluster = rule_compression.new_cluster_from_rule(self.extended_rule)
        self.assertEqual(type(cluster), RuleCluster)
        self.assertEqual(cluster.rules[0], self.extended_rule)

    def test_cluster_rules_hierarchically(self):
        # equal left side value
        self.rule_a.left_side = (('age', 'grandfather'),)
        self.rule_a.all_sides = (('age', 'grandfather'),)
        self.rule_b.left_side = (('age', 'grandfather'), ('haircolor', 'gray'),)
        self.rule_b.all_sides = (('age', 'grandfather'), ('haircolor', 'gray'),)

        self.rule_a.delta_supports_of_left_side = 0.7
        self.rule_a.delta_supports = 0.7
        self.rule_b.delta_supports = 0.7
        self.rule_b.delta_supports_of_left_side = 0.7

        rules = [self.rule_a, self.rule_b]
        rules = sorted(rules, key=lambda x: abs(x.delta_supports_of_left_side))
        length_groups = rule_compression.group_rules_by_length(rules)
        hierarchical_clusters = rule_compression.cluster_rules_hierarchically(length_groups)
        self.assertEqual(len(hierarchical_clusters), 1)

        # subcluster has higher left side value
        self.rule_a.delta_supports_of_left_side = 0.8
        self.rule_a.delta_supports = 0.8
        self.rule_b.delta_supports = 0.7
        self.rule_b.delta_supports_of_left_side = 0.7

        rules = [self.rule_a, self.rule_b]
        rules = sorted(rules, key=lambda x: abs(x.delta_supports_of_left_side))
        length_groups = rule_compression.group_rules_by_length(rules)
        hierarchical_clusters = rule_compression.cluster_rules_hierarchically(length_groups)
        self.assertEqual(len(hierarchical_clusters), 1)

        # subcluster has lower left side value
        self.rule_a.delta_supports_of_left_side = 0.7
        self.rule_a.delta_supports = 0.7
        self.rule_b.delta_supports = 0.9
        self.rule_b.delta_supports_of_left_side = 0.9

        rules = [self.rule_a, self.rule_b]
        rules = sorted(rules, key=lambda x: abs(x.delta_supports_of_left_side))
        length_groups = rule_compression.group_rules_by_length(rules)
        hierarchical_clusters = rule_compression.cluster_rules_hierarchically(length_groups)
        self.assertEqual(len(hierarchical_clusters), 2)


