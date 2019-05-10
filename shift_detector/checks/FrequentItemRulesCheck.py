from shift_detector.checks.Check import Check, CheckResult
from shift_detector.checks.frequent_item_rules import fpgrowth, rule_compression
import pandas as pd


class FrequentItemsetResult(CheckResult):

    def __init__(self, data):
        self.data = data

    def print_report(self):
        """

        Print report for checked columns

        """
        limit = 4
        count = 0
        print(str(limit+1), 'MOST IMPORTANT RULES \n')

        for rule in self.data:
            rule.print()
            count += 1
            if count == limit:
                break


class FrequentItemsetCheck(Check):

    def __init__(self, first_df: pd.DataFrame, second_df: pd.DataFrame):
        Check.__init__(self, first_df, second_df)

    @staticmethod
    def needed_preprocessing():
        return {
            "category": "default",
            "text": "word2vec"
        }

    def set_data(self, dataframes):
        return

    def run(self, columns=[]) -> CheckResult:
        """
        Runs check on provided columns

        :param columns:
        :return: CheckResult

        """
        df1 = self.first_df[columns]
        df2 = self.second_df[columns]
        item_rules = fpgrowth.calculate_frequent_rules(df1, df2)
        compressed_rules = rule_compression.compress_rules(item_rules)
        return FrequentItemsetResult(compressed_rules)
