from unittest import TestCase

from pandas import DataFrame

from shift_detector.checks.conditional_probabilities_check import ConditionalProbabilitiesCheck
from shift_detector.precalculations.store import Store


class TestConditionalProbabilitiesCheck(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.check = ConditionalProbabilitiesCheck()

    def test_run(self):
        report = self.check.run(self.store)
        report.print_report()
        report.print_explanation()
