"""
This package implements an algorithm to compare two data sets using **association rule mining**.

.. _example:

We introduce this algorithm by an example:

    Assume you have two data sets (``ds1`` and ``ds2``) about shoes extracted from a
    product catalog. Each row contains information about *make*, *color* and
    *category*.

    Our algorithm returns a list of *association rules* obeying the following form::

        MAKE: Nike, COLOR: black ==> CATEGORY: football
        [SUPPORTS_OF_LEFT_SIDES: (0.3, 0.07), DELTA_SUPPORTS_OF_LEFT_SIDES: 0.23,
        SUPPORTS: (0.03, 0.05), DELTA_SUPPORTS: -0.02, CONFIDENCES: (0.1, 0.71),
        DELTA_CONFIDENCES: -0.61]

    This rule states that (a) 30% of the tuples in ``ds1`` and 7% of the tuples in
    ``ds2`` are about black Nike shoes, which accounts to a difference of 23%,
    (b) 3% of the tuples in ``ds1`` and 5% of the tuples in ``ds2`` are about black
    Nike football shoes, which accounts to a difference of -2%, and
    (c) if a tuple is about black Nike shoes the **conditional probability** that
    the category is football is 10% in ``ds1`` and 71% in ``ds2``, which
    accounts to a difference of -61%.

    On the basis of such rules, you can easily get insights into differences
    between the two data sets. The above rule alone tells you, that (a) ``ds1``
    contains way more tuples about black Nike shoes than ``ds2``, however, (b) the
    probability that such a shoe is made for football is way higher in ``ds2`` than
    in ``ds1``.

The algorithm proceeds with the following steps:

1. Both data sets are transformed: each component of every tuple is replaced by an
   attribute-name, attribute-value pair. This is required for the correct
   working of the next step. However, the transformation is applied on the fly; we
   never actually copy the data.
2. Our implementation of FP-growth is used to generate *association rules* for both
   data sets individually. The parameters ``relative_min_support`` and
   ``relative_min_confidence`` are used as described in [Han2000]_ and
   [Agrawal1994]_. The only difference is that both parameters are relative and
   expect ``floats`` between 0 and 1, whereas [Han2000]_ and [Agrawal1994]_ use the
   absolute value ``min_support``:

   ``relative_min_support``:
     This parameter mainly impacts the runtime of FP-growth. The lower
     ``relative_min_support`` the more resources are required during computation
     both in terms of memory and CPU. The default value is ``0.01``, which is high
     enough to get a reasonably good performance and still low enough to not
     prematurely exclude significant association rules. This parameter allows you to
     adjust the granularity of the comparison of the two data sets.

   ``relative_min_confidence``:
     This parameter impacts the amount of generated association rules. The higher
     ``relative_min_confidence`` the more rules are generated. The default value is
     ``0.15``. There is no further significance in this value other than that it
     seems sufficiently reasonable.

   Only association rules whose support exceeds ``relative_min_support`` and whose
   confidence exceeds ``relative_min_confidence`` in at least one data set are
   included in the generated association rules.
3. All association rules exceeding ``relative_min_support`` and
   ``relative_min_confidence`` in both data sets can be compared directly. For each
   such rule generate one association rule of the form showed in the example_ above.
4. If a rule exceeds ``relative_min_support`` and ``relative_min_confidence`` in
   one data set but not in the other, we don't know if this rule does not appear in
   the other data set at all or just does not exceed ``relative_min_support`` and/or
   ``relative_min_confidence``. We therefore have to scan both data sets one
   last time to aggregate the counts of such rules. This information at hand, we can
   generate the remaining association rules and our algorithm terminates.

.. [Han2000] Jiawei Han, Jian Pei, and Yiwen Yin. 2000. Mining frequent patterns
   without candidate generation. In Proceedings of the 2000 ACM SIGMOD international
   conference on Management of data (SIGMOD '00). ACM, New York, NY, USA, 1-12
.. [Agrawal1994] Rakesh Agrawal and Ramakrishnan Srikant. 1994. Fast Algorithms for
   Mining Association Rules in Large Databases. In Proceedings of the 20th
   International Conference on Very Large Data Bases (VLDB '94), Jorge B. Bocca,
   Matthias Jarke, and Carlo Zaniolo (Eds.). Morgan Kaufmann Publishers Inc., San
   Francisco, CA, USA, 487-499.
"""
from collections import namedtuple

from shift_detector.checks.frequent_item_rules import pyfpgrowth_core
from collections import namedtuple


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


def calculate_frequent_rules(df1, df2, relative_min_support: float = 0.01, relative_min_confidence: float = 0.15):
    columns = df1.columns
    column_to_index = {c: i for i, c in enumerate(columns)}

    transactions = (DataFrameIteratorAdapter(df1), DataFrameIteratorAdapter(df2))

    absolute_min_supports = (round(relative_min_support * len(transactions[0])),
                             round(relative_min_support * len(transactions[1])))

    patterns = (pyfpgrowth_core.find_frequent_patterns(transactions[0], absolute_min_supports[0]),
                pyfpgrowth_core.find_frequent_patterns(transactions[1], absolute_min_supports[1]))

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
        rule = get_absolute_supports(second_exclusives, transactions[0])
        for key in second_exclusives:
            support = rule[tuple(sorted(key.left_side + key.right_side))] / len(transactions[0])
            support_of_left_side = rule[key.left_side] / len(transactions[0])
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
