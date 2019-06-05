from shift_detector.precalculations.conditional_probabilities.ExtendendRule import ExtendedRule, RuleCluster
from collections import defaultdict


def remove_attribute_value_pairs_appearing_in_all_rules(rules):
    if not rules:
        return []

    appearing_in_all_seen_so_far = set(rules[0].left_side + rules[0].right_side)

    for rule in rules:
        not_appearing_in_this_rule = set()
        for attribute_value_pair in appearing_in_all_seen_so_far:
            if attribute_value_pair not in set(rule.left_side + rule.right_side):
                not_appearing_in_this_rule.add(attribute_value_pair)

        appearing_in_all_seen_so_far -= not_appearing_in_this_rule

    if appearing_in_all_seen_so_far:
        for rule in rules:
            rule.left_side = [attribute_value_pair for attribute_value_pair in rule.left_side if attribute_value_pair not in appearing_in_all_seen_so_far]
            rule.right_side = [attribute_value_pair for attribute_value_pair in rule.right_side if attribute_value_pair not in appearing_in_all_seen_so_far]
    return rules


def transform_to_extended_rules(rules):
    extended_rules = []
    for rule in rules:
        extended_rules.append(ExtendedRule(rule.left_side, rule.right_side, rule.supports_of_left_side,
                                           rule.delta_supports_of_left_side, rule.supports, rule.delta_supports,
                                           rule.confidences, rule.delta_confidences))

    return extended_rules


def group_rules_by_length(rules):
    groupings = defaultdict(list)

    for rule in rules:
        groupings[len(rule.left_side + rule.right_side)].append(rule)

    return groupings


def new_cluster_from_rule(rule):
    rule_attributes = rule.right_side + rule.left_side
    new_cluster = RuleCluster(rule_attributes, [rule])
    new_cluster.max_abs_delta_supports = rule.delta_supports_of_left_side
    return new_cluster


def cluster_rules_hierarchically(length_groups):
    hierarchical_clusters = []
    for group_of_same_length in length_groups:

        for rule in group_of_same_length:

            supercluster_was_found = False
            possible_supercluster = []

            for established_cluster in hierarchical_clusters:
                if established_cluster.is_supercluster(rule):
                    supercluster_was_found = True
                    possible_supercluster.append(established_cluster)

            if supercluster_was_found:
                superclusters = sorted(possible_supercluster, key=lambda x: x.max_abs_delta_supports,
                                       reverse=True)
                highest_support_supercluster = superclusters[0]
                highest_support_supercluster.subcluster.append(rule)

            else:
                new_cluster = new_cluster_from_rule(rule)
                new_cluster.calculate_max_support_and_confidence()
                hierarchical_clusters.append(new_cluster)

    return hierarchical_clusters


def compress_rules(uncompressed_rules):
    transformed_rules = transform_to_extended_rules(uncompressed_rules)
    simplified_rules = remove_attribute_value_pairs_appearing_in_all_rules(transformed_rules)

    sorted_rules = sorted(simplified_rules, key=lambda rule: abs(rule.delta_supports))
    grouped_rules = group_rules_by_length(sorted_rules)
    hierarchical_clusters = cluster_rules_hierarchically(grouped_rules)

    for cluster in hierarchical_clusters:
        cluster.calculate_max_support_and_confidence()

    hierarchical_clusters = sorted(hierarchical_clusters, key=lambda x: x.max_abs_delta_supports,
                                   reverse=True)
    return hierarchical_clusters
