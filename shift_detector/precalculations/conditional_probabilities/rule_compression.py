from collections import defaultdict

from shift_detector.precalculations.conditional_probabilities.ExtendendRule import ExtendedRule, RuleCluster


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
            rule.left_side = [attribute_value_pair for attribute_value_pair in rule.left_side if
                              attribute_value_pair not in appearing_in_all_seen_so_far]
            rule.right_side = [attribute_value_pair for attribute_value_pair in rule.right_side if
                               attribute_value_pair not in appearing_in_all_seen_so_far]
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

    return [groupings[key] for key in sorted(groupings)]


def cluster_rules_hierarchically(grouped_rules):
    clusters = []
    for group in grouped_rules:
        for rule in group:
            highest_support_super_cluster = max((cluster for cluster in clusters if cluster.is_super_cluster_of(rule)),
                                                key=lambda c: abs(c.rule.delta_supports), default=None)
            if highest_support_super_cluster:
                highest_support_super_cluster.sub_clusters.append(rule)
            else:
                clusters.append(RuleCluster(rule))

    return clusters


def compress_rules(uncompressed_rules):
    transformed_rules = transform_to_extended_rules(uncompressed_rules)
    simplified_rules = remove_attribute_value_pairs_appearing_in_all_rules(transformed_rules)

    grouped_rules = group_rules_by_length(simplified_rules)
    hierarchical_clusters = cluster_rules_hierarchically(grouped_rules)

    return sorted(hierarchical_clusters, key=lambda c: abs(c.rule.delta_supports),
                  reverse=True)
