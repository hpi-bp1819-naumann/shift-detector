from collections import defaultdict

from shift_detector.precalculations.conditional_probabilities.ExtendendRule import ExtendedRule, RuleCluster

# TODO
def remove_attribute_value_pairs_appearing_in_all_rules(rules):
    # TODO: remove this function.
    appearing_in_all_seen_so_far = None

    for rule in rules:
        if not appearing_in_all_seen_so_far:
            # initialize appearing_in_all_seen_so_far
            appearing_in_all_seen_so_far = set(rule.left_side + rule.right_side)
            continue
        this_rule = set(rule.left_side + rule.right_side)
        not_appearing_in_this_rule = set(attribute_value_pair for attribute_value_pair in appearing_in_all_seen_so_far
                                         if attribute_value_pair not in this_rule)

        appearing_in_all_seen_so_far -= not_appearing_in_this_rule

    if appearing_in_all_seen_so_far:
        simplified_rules = []
        for rule in rules:
            rule.left_side = [attribute_value_pair for attribute_value_pair in rule.left_side if
                              attribute_value_pair not in appearing_in_all_seen_so_far]
            rule.right_side = [attribute_value_pair for attribute_value_pair in rule.right_side if
                               attribute_value_pair not in appearing_in_all_seen_so_far]
            if rule.left_side:
                simplified_rules.append(rule)
        return simplified_rules
    else:
        return rules


def transform_to_extended_rules(rules):
    return [ExtendedRule(rule.left_side, rule.right_side, rule.supports_of_left_side,
                         rule.delta_supports_of_left_side, rule.supports, rule.delta_supports,
                         rule.confidences, rule.delta_confidences)
            for rule in rules]


def group_rules_by_length(rules):
    groupings = defaultdict(list)

    for rule in rules:
        groupings[len(rule.left_side + rule.right_side)].append(rule)

    return [groupings[group_length] for group_length in sorted(groupings)]


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
    # simplified_rules = remove_attribute_value_pairs_appearing_in_all_rules(transformed_rules)

    grouped_rules = group_rules_by_length(transformed_rules)
    hierarchical_clusters = cluster_rules_hierarchically(grouped_rules)

    return sorted(hierarchical_clusters, key=lambda c: abs(c.rule.delta_supports),
                  reverse=True)
