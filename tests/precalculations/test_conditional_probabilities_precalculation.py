import unittest
from collections import namedtuple
from itertools import chain, combinations

import pandas as pd

from shift_detector.precalculations.conditional_probabilities import fpgrowth, pyfpgrowth_core
from shift_detector.precalculations.conditional_probabilities_precalculation import \
    ConditionalProbabilitiesPrecalculation


class TestFPGrowth(unittest.TestCase):
    def test_find_frequent_patterns(self):
        transactions = [
            [1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]
        ]
        min_support = 2
        result = pyfpgrowth_core.find_frequent_patterns(transactions, min_support)
        expected = {(1,): 2, (2,): 3, (3,): 3, (5,): 3, (1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2, (2, 3, 5): 2}
        self.assertDictEqual(result, expected)

    def test_generate_association_rules(self):
        frequent_items = {(1,): 2, (2,): 3, (3,): 3, (5,): 3, (1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2, (2, 3, 5): 2}
        num_rows = 4
        min_confidence = 0.1

        def get_non_empty_subsets(iterable):
            """Return all non empty subsets of ``iterable`` including ``iterable`` itself."""
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

        Data = namedtuple('Data', ['left_side', 'right_side'])
        MetaData = namedtuple('MetaData', ['support_of_left_side', 'support', 'confidence'])

        expected = {}
        for frequent_item in frequent_items:
            for left_side in get_non_empty_subsets(frequent_item):
                support_of_left_side = frequent_items[left_side]
                right_side = tuple(sorted(set(frequent_item) - set(left_side)))
                data = Data(left_side=left_side, right_side=right_side)
                meta_data = MetaData(support_of_left_side=support_of_left_side / num_rows,
                                     support=frequent_items[frequent_item] / num_rows,
                                     confidence=frequent_items[frequent_item] / support_of_left_side)

                if meta_data.confidence >= min_confidence:
                    expected[data] = meta_data

        result = pyfpgrowth_core.generate_association_rules(frequent_items, min_confidence, num_rows)
        self.assertDictEqual(expected, result)

    def test_iterator(self):
        data = [[1, 2], [3, 4]]
        columns = ['a', 'b']
        df = pd.DataFrame(data, columns=columns)
        it = fpgrowth.DataFrameIteratorAdapter(df)
        self.assertEqual(len(it), 2)
        for i, t in enumerate(it):
            with self.subTest():
                self.assertEqual(t, [(columns[0], data[i][0]), (columns[1], data[i][1])])

        for i, t in enumerate(it):
            with self.subTest():
                self.assertEqual(t, [(columns[0], data[i][0]), (columns[1], data[i][1])])

    def test_calculate_frequent_rules(self):
        columns = ['c1', 'c2']
        data1 = [['exclusive 1', 1], ['common', 42]]
        data2 = [['exclusive 2', 2], ['common', 42]]
        df1 = pd.DataFrame(data1, columns=columns)
        df2 = pd.DataFrame(data2, columns=columns)
        rules = fpgrowth.calculate_frequent_rules(df1, df2, 0.01, 0.15)
        Rule = namedtuple('Rule', ['left_side', 'right_side', 'supports_of_left_side', 'delta_supports_of_left_side',
                                   'supports', 'delta_supports', 'confidences', 'delta_confidences'])
        expected = [Rule(left_side=(('c1', 'common'),), right_side=(), supports_of_left_side=(0.5, 0.5),
                         delta_supports_of_left_side=0.0, supports=(0.5, 0.5), delta_supports=0.0,
                         confidences=(1.0, 1.0), delta_confidences=0.0),
                    Rule(left_side=(('c2', 42),), right_side=(('c1', 'common'),), supports_of_left_side=(0.5, 0.5),
                         delta_supports_of_left_side=0.0, supports=(0.5, 0.5), delta_supports=0.0,
                         confidences=(1.0, 1.0), delta_confidences=0.0),
                    Rule(left_side=(('c1', 'common'), ('c2', 42)), right_side=(), supports_of_left_side=(0.5, 0.5),
                         delta_supports_of_left_side=0.0, supports=(0.5, 0.5), delta_supports=0.0,
                         confidences=(1.0, 1.0), delta_confidences=0.0),
                    Rule(left_side=(('c2', 42),), right_side=(), supports_of_left_side=(0.5, 0.5),
                         delta_supports_of_left_side=0.0, supports=(0.5, 0.5), delta_supports=0.0,
                         confidences=(1.0, 1.0), delta_confidences=0.0),
                    Rule(left_side=(('c1', 'common'),), right_side=(('c2', 42),), supports_of_left_side=(0.5, 0.5),
                         delta_supports_of_left_side=0.0, supports=(0.5, 0.5), delta_supports=0.0,
                         confidences=(1.0, 1.0), delta_confidences=0.0),
                    Rule(left_side=(('c1', 'exclusive 1'), ('c2', 1)), right_side=(), supports_of_left_side=(0.5, 0.0),
                         delta_supports_of_left_side=0.5, supports=(0.5, 0.0), delta_supports=0.5,
                         confidences=(1.0, 0.0), delta_confidences=1.0),
                    Rule(left_side=(('c1', 'exclusive 1'),), right_side=(('c2', 1),), supports_of_left_side=(0.5, 0.0),
                         delta_supports_of_left_side=0.5, supports=(0.5, 0.0), delta_supports=0.5,
                         confidences=(1.0, 0.0), delta_confidences=1.0),
                    Rule(left_side=(('c2', 1),), right_side=(), supports_of_left_side=(0.5, 0.0),
                         delta_supports_of_left_side=0.5, supports=(0.5, 0.0), delta_supports=0.5,
                         confidences=(1.0, 0.0), delta_confidences=1.0),
                    Rule(left_side=(('c1', 'exclusive 1'),), right_side=(), supports_of_left_side=(0.5, 0.0),
                         delta_supports_of_left_side=0.5, supports=(0.5, 0.0), delta_supports=0.5,
                         confidences=(1.0, 0.0), delta_confidences=1.0),
                    Rule(left_side=(('c2', 1),), right_side=(('c1', 'exclusive 1'),), supports_of_left_side=(0.5, 0.0),
                         delta_supports_of_left_side=0.5, supports=(0.5, 0.0), delta_supports=0.5,
                         confidences=(1.0, 0.0), delta_confidences=1.0),
                    Rule(left_side=(('c2', 2),), right_side=(('c1', 'exclusive 2'),), supports_of_left_side=(0.0, 0.5),
                         delta_supports_of_left_side=-0.5, supports=(0.0, 0.5), delta_supports=-0.5,
                         confidences=(0.0, 1.0), delta_confidences=-1.0),
                    Rule(left_side=(('c2', 2),), right_side=(), supports_of_left_side=(0.0, 0.5),
                         delta_supports_of_left_side=-0.5, supports=(0.0, 0.5), delta_supports=-0.5,
                         confidences=(0.0, 1.0), delta_confidences=-1.0),
                    Rule(left_side=(('c1', 'exclusive 2'), ('c2', 2)), right_side=(), supports_of_left_side=(0.0, 0.5),
                         delta_supports_of_left_side=-0.5, supports=(0.0, 0.5), delta_supports=-0.5,
                         confidences=(0.0, 1.0), delta_confidences=-1.0),
                    Rule(left_side=(('c1', 'exclusive 2'),), right_side=(('c2', 2),), supports_of_left_side=(0.0, 0.5),
                         delta_supports_of_left_side=-0.5, supports=(0.0, 0.5), delta_supports=-0.5,
                         confidences=(0.0, 1.0), delta_confidences=-1.0),
                    Rule(left_side=(('c1', 'exclusive 2'),), right_side=(), supports_of_left_side=(0.0, 0.5),
                         delta_supports_of_left_side=-0.5, supports=(0.0, 0.5), delta_supports=-0.5,
                         confidences=(0.0, 1.0), delta_confidences=-1.0)]
        self.assertCountEqual(rules, expected)

    def test_equal_and_hash(self):
        a = ConditionalProbabilitiesPrecalculation(0.5, 0.4)
        b = ConditionalProbabilitiesPrecalculation(0.5, 0.4)
        c = ConditionalProbabilitiesPrecalculation(0.5, 0.5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(hash(a), hash(c))


if __name__ == '__main__':
    unittest.main()
