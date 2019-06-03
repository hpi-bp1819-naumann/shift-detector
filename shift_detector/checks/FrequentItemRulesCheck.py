from collections import defaultdict
import itertools

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.FrequentItemsetPrecalculation import FrequentItemsetPrecalculation
from shift_detector.precalculations.Store import Store


class FrequentItemsetCheck(Check):
    """
    The frequent item set object implements the conditional probability check.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        requiqred during computation both in terms of memory and CPU. The default
        value is ``0.01``, which is high enough to get a reasonably good performance
        and still low enough to not prematurely exclude significant association rules.
        This parameter allows you to adjust the granularity of the comparison of the
        two data sets.
    :param min_confidence: a float between 0 and 1. This parameter impacts the amount
        of generated association rules. The higher ``min_confidence`` the more rules
        are generated. The default value is ``0.15``. There is no further significance
        in this value other than that it seems sufficiently reasonable.
    """

    def __init__(self, min_support=0.01, min_confidence=0.15, rule_limit=5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rule_limit = rule_limit

    def run(self, store: Store) -> Report:
        """
        Calculate frequent rules, compress them and create a
        FrequentItemsetReport.
        :param store: the Store
        :return: FrequentItemsetReport
        """
        precalculation_result = store[FrequentItemsetPrecalculation()]
        compressed_rules = precalculation_result['compressed_rules']
        examined_columns = precalculation_result['examined_columns']

        shifted_clumns = list()
        information = defaultdict(list)
        count = 0
        for rulecluster in compressed_rules:
            columns = []
            for (key, val) in rulecluster.attributes:
                columns.append(key)
            shifted_clumns.append(columns)

            information[str(columns)].append(rulecluster.__str__())
            count += 1
            if count == self.rule_limit:
                break

        # remove duplicates from list of lists
        shifted_clumns.sort()
        shifted_clumns = list(shifted_clumns for shifted_clumns, _ in itertools.groupby(shifted_clumns))

        return Report(examined_columns, shifted_clumns, information)
