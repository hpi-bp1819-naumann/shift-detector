from shift_detector.checks.frequent_item_rules.ExtendendRule import ExtendedRule, RuleCluster
# from shift_detector.checks.frequent_item_rules import fpgrowth


def printrule(rule):
    if len(rule.left_side) > 0:
        print(rule.left_side, '-->', rule.right_side)
        print('supports_left: \t', rule.supports_of_left_side[0], ' - ', rule.supports_of_left_side[1],
              '=', rule.delta_supports_of_left_side)
    else:
        print('//rule has no left side')
        print('() -->', rule.right_side)
    print()
    return


def remove_allsame_attributes(rules):
    """
    Given a list of rules, return the same list but having removed those attributes that are the same across all
    elements in the list
    """

    if len(rules) == 0:
        return []
    duplicate_candidates = set(rules[0].right_side + rules[0].left_side)

    for rule in rules:
        non_duplicates = []
        for duplicate_candidate in duplicate_candidates:
            if duplicate_candidate not in set(rule.left_side + rule.right_side):
                non_duplicates.append(duplicate_candidate)

        for non_duplicate in non_duplicates:
            duplicate_candidates.remove(non_duplicate)

    for i, value in enumerate(rules):
        left_side = [t for t in rules[i].left_side if t not in duplicate_candidates]
        right_side = [t for t in rules[i].right_side if t not in duplicate_candidates]

        rules[i] = ExtendedRule(left_side, right_side, rules[i].supports_of_left_side,
                                rules[i].delta_supports_of_left_side, rules[i].supports, rules[i].delta_supports,
                                rules[i].confidences, rules[i].delta_confidences)

    return rules


def true_if_no_attribute_none(rule):
    for attribute in rule.all_sides():
        if attribute[1] is None:
            return False
    return True


def filter_non_values(rules):
    rules = list(filter(true_if_no_attribute_none, rules))
    return rules


def add_side_attributes_to_rules(rules):
    rules_extended = []
    for rule in rules:
        rule_ext = ExtendedRule(rule.left_side, rule.right_side, rule.supports_of_left_side,
                                rule.delta_supports_of_left_side, rule.supports, rule.delta_supports,
                                rule.confidences, rule.delta_confidences)
        rules_extended.append(rule_ext)

    return rules_extended


def group_rules_by_length(rules):
    groupings = []
    for i in range(10):
        groupings.append([])

    for rule in rules:
        groupings[len(rule.all_sides())].append(rule)

    groupings_cleaned = [x for x in groupings if x != []]
    return groupings_cleaned


def new_cluster_from_rule(rule):
    rule_attributes = rule.right_side + rule.left_side
    new_cluster = RuleCluster(rule_attributes, [rule])
    new_cluster.max_abs_delta_support_left = rule.delta_supports_of_left_side
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
                superclusters = sorted(possible_supercluster, key=lambda x: x.max_abs_delta_support_left,
                                       reverse=True)
                highest_support_supercluster = superclusters[0]
                highest_support_supercluster.subcluster.append(rule)

            else:
                new_cluster = new_cluster_from_rule(rule)
                new_cluster.calculate_max_support_and_confidence()
                hierarchical_clusters.append(new_cluster)

    return hierarchical_clusters


def compress_rules(rules):
    rules = add_side_attributes_to_rules(rules)
    rules = remove_allsame_attributes(rules)
    rules = filter_non_values(rules)

    rules = sorted(rules, key=lambda x: abs(x.delta_supports_of_left_side))
    length_groups = group_rules_by_length(rules)
    hierarchical_clusters = cluster_rules_hierarchically(length_groups)

    for cluster in hierarchical_clusters:
        cluster.calculate_max_support_and_confidence()

    hierarchical_clusters = sorted(hierarchical_clusters, key=lambda x: x.max_abs_delta_support_left,
                                   reverse=True)
    return hierarchical_clusters


# if __name__ == '__main__':
#     compress_rules()







