from unittest import TestCase

from pandas import DataFrame

from Morpheus.checks.conditional_probabilities_check import ConditionalProbabilitiesCheck
from Morpheus.precalculations.store import Store


class TestConditionalProbabilitiesCheck(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.check = ConditionalProbabilitiesCheck()

    def test_run(self):
        self.check.run(self.store)
