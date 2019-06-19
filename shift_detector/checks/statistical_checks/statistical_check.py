from abc import abstractmethod

import pandas as pd
from IPython.display import display

from shift_detector.checks.check import Check, Report


class StatisticalCheck(Check):
    """
    Blueprint for a statistical test check.
    :param significance: columns which are equally distributed with a probability lower than this are marked as shifted
    :param sampling: whether or not to use sampling for the larger set if compared data sets have unequal sizes
    :param sampling_seed: seed to use for sampling, if sampling is enabled
    """
    def __init__(self, significance=0.01, use_sampling=False, sampling_seed=0):
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
    def data_to_process(self, store):
        """
        Receive the data to run on.
        :return: Processed df1, df2 and the columns
        """
        return pd.DataFrame(), pd.DataFrame(), []

    def explain(self, pvalues):
        """
        Generates a dictionary with textual explanations for all significant columns.
        :param pvalues: dataframe with the test result of each column
        :return: dictionary of column-explanation-pairs
        """
        explanations = {}
        for column in self.significant_columns(pvalues):
            explanations[column] = '- probability for equal distribution p = {pvalue}\n' \
                                   '- specified significance level alpha = {significance}\n' \
                                   '- statistical test performed: {test_name}'.format(
                                        pvalue=str(pvalues[column]),
                                        significance=str(self.significance),
                                        test_name=self.statistical_test_name()
                                    )
        return explanations

    @staticmethod
    @abstractmethod
    def column_figure(column, df1, df2):
        pass

    def column_figures(self, significant_columns, df1, df2):
        plot_functions = []
        for column in significant_columns:
            plot_functions.append(lambda col=column: self.column_figure(col, df1, df2))
        return plot_functions

    def run(self, store) -> Report:
        pvalues = pd.DataFrame(index=['pvalue'])

        df1, df2, columns = self.data_to_process(store)

        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.use_sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.use_sampling else df2

        for column in columns:
            p = self.statistical_test(part1[column], part2[column])
            pvalues[column] = [p]
        significant_columns = self.significant_columns(pvalues)
        return StatisticalReport("Statistical Check",
                                 examined_columns=columns,
                                 shifted_columns=significant_columns,
                                 explanation=self.explain(pvalues),
                                 information={'test_results': pvalues},
                                 figures=self.column_figures(significant_columns, part1, part2))


class StatisticalReport(Report):

    def print_information(self):
        test_results = self.information['test_results']
        print("Test Result:")
        display(test_results)
