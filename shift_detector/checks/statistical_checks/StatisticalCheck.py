from abc import abstractmethod

import pandas as pd

from shift_detector.checks.Check import Check, Report


class StatisticalCheck(Check):
    """
    Blueprint for a statistical test check.
    :param significance: columns which are equally distributed with a probability lower than this are marked as shifted
    :param sampling: whether or not to use sampling for the larger set if compared data sets have unequal sizes
    :param sampling_seed: seed to use for sampling, if sampling is enabled
    """
    def __init__(self, significance=0.01, use_sampling=False, sampling_seed=None):
        self.significance = significance
        self.use_sampling = use_sampling
        self.seed = sampling_seed

    def is_significant(self, p: float) -> bool:
        """
        Returns True if p-value is smaller or equal to the specified significance level.
        :param p: probability to check significance for
        :return: True or False
        """
        return p <= self.significance

    def significant_columns(self, pvalues):
        """
        Returns True if p-value is smaller or equal to the specified significance level.
        :param p: probability to check significance for
        :return: True or False
        """
        return set(column for column in pvalues.columns if self.is_significant(pvalues.loc['pvalue', column]))

    @abstractmethod
    def run(self, store) -> Report:
        pass


class SimpleStatisticalCheck(StatisticalCheck):
    """
    Blueprint for a statistical test check which involves only one kind of statistical test
    """
    @abstractmethod
    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        """
        Performs a statistical test on the two samples and returns probability that they originate from the same
        distribution aka the p-value of the test.
        :param part1: first sample
        :param part2: second sample
        :return: p-value of statistical test
        """
        pass

    @abstractmethod
    def statistical_test_name(self) -> str:
        """
        Returns the name of the statistical test performed in statistical_test as String.
        :return: name of the statistical test
        """
        pass

    @abstractmethod
    def store_keys(self):
        """
        Returns a list of store keys to retrieve the data with all required precalculations.
        :return: List[ColumnType or Precalculation]
        """
        return []

    def explain(self, pvalues):
        """
        Generates a dictionary with textual explanations for all significant columns.
        :param pvalues: dataframe with the test result of each column
        :return: dictionary of column-explanation-pairs
        """
        explanation = {}
        for column in self.significant_columns(pvalues):
            explanation[column] = 'The probability for equal distribution of the column ' + str(column) + \
                                  ' in both datasets is p = ' + str(pvalues[column]) + ', which is lower than the' \
                                  'specified significance level of alpha = ' + str(self.significance) + '. The ' \
                                  'statistical test performed was ' + self.statistical_test_name() + '.'
        return explanation

    def run(self, store) -> Report:
        pvalues = pd.DataFrame(index=['pvalue'])
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        for columns1, columns2 in [store[key] for key in self.store_keys()]:
            df1 = pd.concat([df1, columns1])
            df2 = pd.concat([df2, columns2])
        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.use_sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.use_sampling else df2
        for column in df1.columns:
            p = self.statistical_test(part1[column], part2[column])
            pvalues[column] = [p]
        return Report(examined_columns=list(df1.columns),
                      shifted_columns=self.significant_columns(pvalues),
                      explanation=self.explain(pvalues),
                      information={'test_results': pvalues})
