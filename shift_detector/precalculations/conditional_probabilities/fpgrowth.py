from collections import namedtuple

from shift_detector.precalculations.conditional_probabilities import pyfpgrowth_core


class DataFrameIteratorAdapter:
    def __init__(self, df):
        self.df = df

    def iterator_factory(self):
        for _, row in self.df.iterrows():
            yield [(c, row[c]) for c in self.df.columns]

    def __iter__(self):
        return self.iterator_factory()

    def __len__(self):
        return len(self.df)


def to_string(rule):
    return '{left_sides} => {right_sides} [SLS: ({sols}), S: ({supports}), C: ({confidences})]'.format(
        left_sides=', '.join('{}: {}'.format(attr, val) for attr, val in rule.left_side),
        right_sides='()' if not rule.right_side else ', '.join(
            '{}: {}'.format(attr, val) for attr, val in rule.right_side),
        sols=', '.join('{:.0%}'.format(val) for val in rule.supports_of_left_side),
        supports=', '.join('{:.0%}'.format(val) for val in rule.supports),
        confidences=', '.join('{:.0%}'.format(val) for val in rule.confidences)
    )


def calculate_frequent_rules(df1, df2, min_support, min_confidence):
    columns = df1.columns
    column_to_index = {c: i for i, c in enumerate(columns)}

    transactions = (DataFrameIteratorAdapter(df1), DataFrameIteratorAdapter(df2))

    absolute_min_supports = (round(min_support * len(transactions[0])),
                             round(min_support * len(transactions[1])))

    patterns = (pyfpgrowth_core.find_frequent_patterns(transactions[0], absolute_min_supports[0]),
                pyfpgrowth_core.find_frequent_patterns(transactions[1], absolute_min_supports[1]))

    rules = (pyfpgrowth_core.generate_association_rules(patterns[0], min_confidence, len(transactions[0])),
             pyfpgrowth_core.generate_association_rules(patterns[1], min_confidence, len(transactions[1])))

    Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                               'supports', 'delta_supports', 'confidences', 'delta_confidences'])

    result = []
    intersection = rules[0].keys() & rules[1].keys()
    # compare rules that exceed min support in both data sets
    for key in intersection:
        result.append(Rule(key.left_side, key.right_side, (rules[0][key].support_of_left_side,
                                                           rules[1][key].support_of_left_side),
                           (rules[0][key].support_of_left_side - rules[1][key].support_of_left_side),
                           (rules[0][key].support, rules[1][key].support),
                           (rules[0][key].support - rules[1][key].support),
                           (rules[0][key].confidence, rules[1][key].confidence),
                           (rules[0][key].confidence - rules[1][key].confidence)))

    def get_absolute_supports(exclusives, other_transactions):
        """Calculate and return absolute support of rules of `exclusives` in `other_transactions`"""
        grouping_attributes = set()
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
        rule = get_absolute_supports(first_exclusives, transactions[1])
        for key in first_exclusives:
            support = rule[tuple(sorted(key.left_side + key.right_side))] / len(transactions[1])
            support_of_left_side = rule[key.left_side] / len(transactions[1])
            if support_of_left_side:
                confidence = support / support_of_left_side
            else:
                confidence = 0.0
            result.append(Rule(
                key.left_side, key.right_side, (rules[0][key].support_of_left_side,
                                                support_of_left_side),
                (rules[0][key].support_of_left_side - support_of_left_side),
                (rules[0][key].support, support), (rules[0][key].support - support),
                (rules[0][key].confidence, confidence), (rules[0][key].confidence - confidence)
            ))

    second_exclusives = rules[1].keys() - rules[0].keys()
    # compare rules exceeding min support only in the second data set
    if second_exclusives:
        rule = get_absolute_supports(second_exclusives, transactions[0])
        for key in second_exclusives:
            support = rule[tuple(sorted(key.left_side + key.right_side))] / len(transactions[0])
            support_of_left_side = rule[key.left_side] / len(transactions[0])
            if support_of_left_side:
                confidence = support / support_of_left_side
            else:
                confidence = 0.0
            result.append(Rule(
                key.left_side, key.right_side, (support_of_left_side,
                                                rules[1][key].support_of_left_side),
                (support_of_left_side - rules[1][key].support_of_left_side),
                (support, rules[1][key].support), (support - rules[1][key].support),
                (confidence, rules[1][key].confidence), (confidence - rules[1][key].confidence)
            ))

    return result
