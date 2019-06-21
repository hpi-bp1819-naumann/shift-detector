from collections import defaultdict

from morpheus.precalculations.conditional_probabilities.rule_cluster import RuleCluster


def group_rules_by_length(rules):
    groupings = defaultdict(list)

    for rule in rules:
        groupings[len(rule.left_side + rule.right_side)].append(rule)

    return (groupings[group_length] for group_length in sorted(groupings))


def cluster_rules_hierarchically(grouped_rules):
    clusters = []
    for group in grouped_rules:
        for rule in group:
            highest_support_super_cluster = max((cluster for cluster in clusters if cluster.is_super_cluster_of(rule)),
                                                key=lambda c: abs(c.main_rule.delta_supports), default=None)
            if highest_support_super_cluster:
                highest_support_super_cluster.sub_rules.append(rule)
            else:
                clusters.append(RuleCluster(rule))

    return clusters


def compress_and_sort_rules(rules):
    grouped_rules = group_rules_by_length(rules)
    clustered_rules = cluster_rules_hierarchically(grouped_rules)

    return sorted(clustered_rules, key=lambda c: abs(c.main_rule.delta_supports), reverse=True)
