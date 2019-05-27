from abc import abstractmethod

import pandas as pd

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


class SimpleStatisticalCheck(Check):
    """
    Blueprint for a statistical test check. Subclasses are runnable checks
    :param sampling: whether or not to use sampling for the larger set if compared data sets have unequal sizes
    :param sampling_seed: seed to use for sampling, if sampling is enabled
    """
    def __init__(self, sampling=False, sampling_seed=None):
        self.sampling = sampling
        self.seed = sampling_seed

    @abstractmethod
    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return 0.0

    @abstractmethod
    def store_keys(self):
        return []

    def run(self, store) -> StatisticalReport:
        pvalues = pd.DataFrame(index=['pvalue'])
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        for columns1, columns2 in [store[key] for key in self.store_keys()]:
            df1 = pd.concat([df1, columns1])
            df2 = pd.concat([df2, columns2])
        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.sampling else df2
        for column in store.columns:
            p = self.statistical_test(part1[column], part2[column])
            pvalues[column] = [p]
        return StatisticalReport(pvalues)