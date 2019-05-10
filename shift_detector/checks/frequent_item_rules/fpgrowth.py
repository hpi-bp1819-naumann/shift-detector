from collections import namedtuple

from shift_detector.checks.frequent_item_rules import pyfpgrowth_core


def calculate_frequent_rules(df1, df2, relative_min_support: float = 0.01, relative_min_confidence: float = 0.3):
    """Use fp-growth algorithm to calculate and compare conditional probabilities."""
    transactions = tuple([] for _ in range(2))
    columns = df1.columns
    column_to_index = {c: i for i, c in enumerate(columns)}

    for index, row in df1.iterrows():
        transactions[0].append([(c, row[c]) for c in columns])

    for index, row in df2.iterrows():
        transactions[1].append([(c, row[c]) for c in columns])

    absolute_min_supports = tuple(round(relative_min_support * len(transactions[i])) for i in range(2))

    patterns = tuple(
        pyfpgrowth_core.find_frequent_patterns(transactions[i], absolute_min_supports[i]) for i in range(2))

    rules = (pyfpgrowth_core.generate_association_rules(patterns[0], relative_min_confidence, len(transactions[0])),
             pyfpgrowth_core.generate_association_rules(patterns[1], relative_min_confidence, len(transactions[1])))

    Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                               'supports', 'delta_supports', 'confidences', 'delta_confidences'])

    result = []
    intersection = rules[0].keys() & rules[1].keys()
    # compare rules that exceed min support in both data sets
    for key in intersection:
        result.append(
            Rule(key.left_side, key.right_side, (rules[0][key].support_of_left_side,
                                                 rules[1][key].support_of_left_side),
                 (rules[0][key].support_of_left_side - rules[1][key].support_of_left_side),
                 (rules[0][key].support, rules[1][key].support),
                 (rules[0][key].support - rules[1][key].support),
                 (rules[0][key].confidence, rules[1][key].confidence),
                 (rules[0][key].confidence - rules[1][key].confidence))
        )

    def get_absolute_supports(exclusives, grouping_attributes, other_transactions):
        """Calculate and return absolute support of rules of `exclusives` in `other_transactions`"""
        rule = {}
        for left_side, right_side in exclusives:
            rule[left_side] = 0
            rule[tuple(sorted(left_side + right_side))] = 0
            grouping_attributes.add(tuple(key for key, value in left_side))
            grouping_attributes.add(tuple(sorted(key for key, value in left_side + right_side)))
        for transaction in other_transactions:
            for group in grouping_attributes:
                indexes = [column_to_index[attr] for attr in group]
                possible_key = tuple(transaction[i] for i in indexes)
                if possible_key in rule:
                    rule[possible_key] += 1
        return rule

    first_exclusives = rules[0].keys() - rules[1].keys()
    # compare rules exceeding min support only in the first data set
    if first_exclusives:
        grouping_attributes = set()
        other_transactions = transactions[1]
        rule = get_absolute_supports(first_exclusives, grouping_attributes, other_transactions)

        for key in first_exclusives:
            support = rule[tuple(sorted(key.left_side + key.right_side))] / len(other_transactions)
            support_of_left_side = rule[key.left_side] / len(other_transactions)
            if support_of_left_side:
                confidence = support / support_of_left_side
            else:
                confidence = 0.0
            result.append(
                Rule(
                    key.left_side, key.right_side, (rules[0][key].support_of_left_side,
                                                    support_of_left_side),
                    (rules[0][key].support_of_left_side - support_of_left_side),
                    (rules[0][key].support, support), (rules[0][key].support - support),
                    (rules[0][key].confidence, confidence), (rules[0][key].confidence - confidence)
                )
            )

    second_exclusives = rules[1].keys() - rules[0].keys()
    # compare rules exceeding min support only in the second data set
    if second_exclusives:
        grouping_attributes = set()
        other_transactions = transactions[0]
        rule = get_absolute_supports(second_exclusives, grouping_attributes, other_transactions)

        for key in second_exclusives:
            support = rule[tuple(sorted(key.left_side + key.right_side))] / len(other_transactions)
            support_of_left_side = rule[key.left_side] / len(other_transactions)
            if support_of_left_side:
                confidence = support / support_of_left_side
            else:
                confidence = 0.0
            result.append(
                Rule(
                    key.left_side, key.right_side, (support_of_left_side,
                                                    rules[1][key].support_of_left_side),
                    (support_of_left_side - rules[1][key].support_of_left_side),
                    (support, rules[1][key].support), (support - rules[1][key].support),
                    (confidence, rules[1][key].confidence), (confidence - rules[1][key].confidence)
                )
            )
    return result
