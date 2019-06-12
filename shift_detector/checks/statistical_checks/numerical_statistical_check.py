import pandas as pd
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
        for data, color in [(df1[column], 'blue'), (df2[column], 'green')]:
            mn, mx = min(pd.concat([df1[column], df2[column]])), max(pd.concat([df1[column], df2[column]])) + 1
            bins = range(mn, mx, int((mx - mn) / 40))
            plt.hist(data, bins=bins, color=color)
            plt.show()