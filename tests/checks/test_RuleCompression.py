import copy
import unittest
from collections import namedtuple

from shift_detector.precalculations.conditional_probabilities import rule_compression
from shift_detector.precalculations.conditional_probabilities.rule_cluster import RuleCluster


class TestRuleCompression(unittest.TestCase):

    def setUp(self):
        Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                                   'supports', 'delta_supports', 'confidences', 'delta_confidences'])
        self.rule_a = Rule(left_side=(('attr1', 'a'),), right_side=(), supports_of_left_side=(0.5, 0.4),
                           delta_supports_of_left_side=0.1, supports=(0.4, 0.1), delta_supports=0.3,
                           confidences=(0.8, 0.25), delta_confidences=0.55)
        self.rule_b = Rule(left_side=(('attr1', 'a'),), right_side=(('attr2', 'b'),), supports_of_left_side=(0.4, 0.4),
                           delta_supports_of_left_side=0, supports=(0.4, 0.4), delta_supports=0,
                           confidences=(1.0, 1.0), delta_confidences=0)
        self.rule_c = Rule(left_side=(('attr1', 'b'),), right_side=(), supports_of_left_side=(0.4, 0.4),
                           delta_supports_of_left_side=0, supports=(0.4, 0.4), delta_supports=0,
                           confidences=(1.0, 1.0), delta_confidences=0)

    def test_supercluster(self):
        cluster_a = RuleCluster(self.rule_a)
        cluster_b = RuleCluster(self.rule_b)

        self.assertTrue(cluster_a.is_super_cluster_of(self.rule_b))
        self.assertFalse(cluster_b.is_super_cluster_of(self.rule_a))

    def test_group_rules_by_length(self):
        grouped_rules = list(rule_compression.group_rules_by_length([self.rule_a, self.rule_b]))
        self.assertEqual(len(grouped_rules), 2)
        self.assertEqual(grouped_rules[0][0], self.rule_a)
        self.assertEqual(grouped_rules[1][0], self.rule_b)

    def test_cluster_rules_hierarchically(self):
        grouped_rules = list(rule_compression.group_rules_by_length([self.rule_a, self.rule_b]))
        clustered_rules = rule_compression.cluster_rules_hierarchically(grouped_rules)
        self.assertEqual(len(clustered_rules), 1)
        self.assertEqual(clustered_rules[0].rule, self.rule_a)
        self.assertEqual(len(clustered_rules[0].sub_rules), 1)
        self.assertEqual(clustered_rules[0].sub_rules[0], self.rule_b)

    def test_compress_rules(self):
        compressed_rules = rule_compression.compress_rules([self.rule_a, self.rule_b, self.rule_c])
        self.assertEqual(len(compressed_rules), 2)
        self.assertGreaterEqual(compressed_rules[0].rule.delta_supports, compressed_rules[1].rule.delta_supports)

