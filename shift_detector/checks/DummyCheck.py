from pandas import DataFrame

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.DummyPreprocessor import DummyPreprocessor


class DummyReport(Report):

    def __init__(self, diff: DataFrame, significance=1000):
        self.diff = diff
        self.significance = significance

    def print_report(self):
        print("Dummy Report")
        print(self.diff)

        for column, column_diff in self.diff.iteritems():
            if column_diff > self.significance:
                print("SHIFT: Mean of {} increased by {}".format(column, column_diff))
            else:
                print("Mean of {} increased by {}".format(column, column_diff))


class DummyCheck(Check):

    def run(self, store) -> DummyReport:
        processed_df1, processed_df2 = store[DummyPreprocessor(5)]

        means1 = processed_df1.mean()
        means2 = processed_df2.mean()

        diff = means2 - means1
        diff = diff.abs()

        return DummyReport(diff)
