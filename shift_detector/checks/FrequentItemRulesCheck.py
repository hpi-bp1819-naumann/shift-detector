from shift_detector.checks.Check import Check, Report
from shift_detector.checks.frequent_item_rules import fpgrowth, rule_compression
from shift_detector.precalculations.Store import Store


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

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.15):
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
