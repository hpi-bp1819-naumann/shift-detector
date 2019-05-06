import csv
from collections import namedtuple
from shift_detector.analyzers.conditional_probabilities import pyfpgrowth_core

relative_min_support = 0.01
min_confidence = 0.01


def main():
    transactions = ([], [])

    columns = [
        'marketplace_id',
        'asin',
        'attribute',
        'refinement_id',
        'item_name',
        'bullet_points',
        'value'
    ]

    column_to_index = {c: i for i, c in enumerate(columns)}

    for i, path in enumerate(['/Users/pzimme/Desktop/Datasets/audits_leonard.csv',
                              '/Users/pzimme/Desktop/Datasets/train_leonard.csv']):

        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, columns, delimiter=',', quotechar='"')
            next(reader)  # skip header line
            for row in reader:
                transactions[i].append([(c, row[c]) for c in columns])

    absolute_min_supports = (round(relative_min_support * len(transactions[0])),
                             round(relative_min_support * len(transactions[1])))

    patterns = (pyfpgrowth_core.find_frequent_patterns(transactions[0], absolute_min_supports[0]),
                pyfpgrowth_core.find_frequent_patterns(transactions[1], absolute_min_supports[1]))

    rules = (pyfpgrowth_core.generate_association_rules(patterns[0], min_confidence, len(transactions[0])),
             pyfpgrowth_core.generate_association_rules(patterns[1], min_confidence, len(transactions[1])))

    Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                               'supports', 'delta_supports', 'confidences', 'delta_confidences'])

    result = []
    intersection = rules[0].keys() & rules[1].keys()
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

    def calculate_support(exclusives, grouping_attributes, rule, other_transactions):
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

    first_exclusives = rules[0].keys() - rules[1].keys()
    if first_exclusives:
        exclusives = first_exclusives
        grouping_attributes = set()
        rule = {}
        other_transactions = transactions[1]
        calculate_support(exclusives, grouping_attributes, rule, other_transactions)

        for key in exclusives:
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
    if second_exclusives:
        exclusives = second_exclusives
        grouping_attributes = set()
        rule = {}
        other_transactions = transactions[0]
        calculate_support(exclusives, grouping_attributes, rule, other_transactions)

        for key in exclusives:
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


if __name__ == '__main__':
    result = main()
