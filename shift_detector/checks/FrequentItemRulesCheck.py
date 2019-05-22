from shift_detector.checks.Check import Check, Report
from shift_detector.checks.frequent_item_rules import fpgrowth, rule_compression
from shift_detector.preprocessors.Store import Store


class FrequentItemsetReport(Report):

    def __init__(self, compressed_rules):
        self.compressed_rules = compressed_rules

    def print_report(self):
        """
        Print report for checked columns
        """
        limit = 4
        count = 0
        print(str(limit+1), 'MOST IMPORTANT RULES \n')

        for rule in self.compressed_rules:
            rule.print()
            count += 1
            if count == limit:
                break


class FrequentItemsetCheck(Check):
    """
    The frequent item set object implements the conditional probability check.

    :param min_support: a float between 0 and 1. This parameter mainly impacts
        the runtime of the check. The lower ``min_support`` the more resources are
        required during computation both in terms of memory and CPU. The default
        value is ``0.01``, which is high enough to get a reasonably good performance
        and still low enough to not prematurely exclude significant association rules.
        This parameter allows you to adjust the granularity of the comparison of the
        two data sets.
    :param min_confidence: a float between 0 and 1. This parameter impacts the amount
        of generated association rules. The higher ``min_confidence`` the more rules
        are generated. The default value is ``0.15``. There is no further significance
        in this value other than that it seems sufficiently reasonable.
    """

    def __init__(self, min_support=0.01, min_confidence=0.15):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def run(self, store: Store) -> FrequentItemsetReport:
        """
        Calculate frequent rules, compress them and create a
        FrequentItemsetReport.
        :param store: the Store
        :return: FrequentItemsetReport
        """
        df1 = store.df1
        df2 = store.df2

        item_rules = fpgrowth.calculate_frequent_rules(df1, df2, self.min_support,
                                                       self.min_confidence)
        compressed_rules = rule_compression.compress_rules(item_rules)
        return FrequentItemsetReport(compressed_rules)
