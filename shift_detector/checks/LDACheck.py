import pandas as pd
from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.LDAEmbedding import LDAEmbedding
from collections import Counter


class LDAReport(Report):

    def __init__(self, diff: pd.DataFrame, significance=1000):
        self.diff = diff
        self.significance = significance

    def print_report(self):
        print("LDA Report")
        print(self.diff)

        for column, column_diff in self.diff.iteritems():
            if column_diff > self.significance:
                print("SHIFT: Mean of {} increased by {}".format(column, column_diff))
            else:
                print("Mean of {} increased by {}".format(column, column_diff))


class LDACheck(Check):

    def run(self, store) -> LDAReport:
        processed_df1, processed_df2 = store[LDAEmbedding()]

        count_topics1 = Counter(processed_df1['topic'])
        count_topics2 = Counter(processed_df2['topic'])

        diff = []


        diff = means2 - means1
        diff = diff.abs()

        return LDAReport(diff)
