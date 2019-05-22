from shift_detector.checks.Check import Check, Report


class StatisticalReport(Report):

    def __init__(self, check_result, significance=0.01):
        self.result = check_result
        self.significance = significance

    def is_significant(self, p: float) -> bool:
        return p <= self.significance

    def significant_columns(self):
        return set(column for column in self.result.columns if self.is_significant(self.result.loc['pvalue', column]))

    def print_report(self):
        print('Columns with probability for equal distribution below significance level ', self.significance, ': ')
        print(self.significant_columns())


class StatisticalCheck(Check):
    pass