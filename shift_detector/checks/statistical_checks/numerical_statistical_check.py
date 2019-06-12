import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.utils.column_management import ColumnType


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(SimpleStatisticalCheck):

    def statistical_test_name(self) -> str:
        return 'Kolmogorov-Smirnov-Two-Sample-Test'

    def store_keys(self):
        return [ColumnType.numerical]

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return kolmogorov_smirnov_test(part1, part2)

    def column_figure(self, column, df1, df2):
        _, bins, _ = plt.hist(df1[column], bins=100, color='cornflowerblue', cumulative=True, histtype='step')
        plt.hist(df2[column], bins=bins, alpha=0.5, color='seagreen', cumulative=True, histtype='step')
        plt.legend([column + ' 1', column + ' 2'], fontsize='x-small')
        plt.title(column + '(Cumulative Distribution)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('number of rows', fontsize='medium')
        plt.show()
        _, bins, _ = plt.hist(df1[column], bins=40, color='cornflowerblue')
        plt.hist(df2[column], bins=bins, alpha=0.5, color='seagreen')
        plt.legend([column + ' 1', column + ' 2'], fontsize='x-small')
        plt.title(column + '(Histogram)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('number of rows', fontsize='medium')
        plt.show()
