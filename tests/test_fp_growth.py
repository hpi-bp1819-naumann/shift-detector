import unittest
from itertools import chain, combinations
from shift_detector.checks.frequent_item_rules import fpgrowth, pyfpgrowth_core
from collections import namedtuple


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


if __name__ == '__main__':
    unittest.main()
