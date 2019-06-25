from abc import abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import gridspec

from shift_detector.checks.check import Check, Report
from shift_detector.utils.visualization import PLOT_GRID_WIDTH, PLOT_ROW_HEIGHT


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
    def check_name(self) -> str:
        """
        Returns the name of the check as String.
        :return: name of the check
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
            explanations[column] = 'p = {pvalue}\n'.format(
                                        pvalue=str(pvalues.loc['pvalue', column])
                                    )
        return explanations

    @staticmethod
    @abstractmethod
    def column_plot(figure, tile, column, df1, df2):
        pass

    @abstractmethod
    def number_of_columns_of_plots(self) -> int:
        pass

    def plot_all_columns(self, plot_functions):
        cols = self.number_of_columns_of_plots()
        rows = int(np.ceil(len(plot_functions) / cols))
        fig = plt.figure(figsize=(PLOT_GRID_WIDTH, PLOT_ROW_HEIGHT * rows), tight_layout=True)
        grid = gridspec.GridSpec(rows, cols)
        for plot_function, tile in zip(plot_functions, grid):
            plot_function(fig, tile)
        plt.show()

    def plot_functions(self, significant_columns, df1, df2):
        plot_functions = []
        for column in sorted(significant_columns):
            plot_functions.append(lambda figure, tile, col=column: self.column_plot(figure, tile, col, df1, df2))
        return plot_functions

    def column_figure(self, significant_columns, df1, df2):
        if not significant_columns:
            return []
        return [lambda plots=tuple(self.plot_functions(significant_columns, df1, df2)): self.plot_all_columns(plots)]

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
        header = 'Performed statistical test: {test_name}\n'.format(test_name=self.statistical_test_name()) + \
                 'Significance level: {significance}\n'.format(significance=str(self.significance))
        return StatisticalReport(self.check_name(),
                                 examined_columns=columns,
                                 shifted_columns=significant_columns,
                                 explanation=self.explain(pvalues),
                                 explanation_header=header,
                                 information={'test_results': pvalues},
                                 figures=self.column_figure(significant_columns, part1, part2))


class StatisticalReport(Report):

    def __init__(self, check_name, examined_columns, shifted_columns, explanation={}, explanation_header=None,
                 information={}, figures=[]):
        super().__init__(check_name, examined_columns, shifted_columns, explanation, information, figures)
        self.explanation_header = explanation_header

    def print_explanation(self):
        print(self.explanation_header)
        super().print_explanation()

    def print_information(self):
        test_results = self.information['test_results']
        print("p-values of all columns:")
        display(test_results)
