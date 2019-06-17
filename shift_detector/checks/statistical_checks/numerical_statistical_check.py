import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.utils.column_management import ColumnType
from shift_detector.utils.visualization import plot_cumulative_step_ratio_histogram, plot_ratio_histogram


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(SimpleStatisticalCheck):

    def statistical_test_name(self) -> str:
        return 'Kolmogorov-Smirnov-Two-Sample-Test'

    def data_to_process(self, store):
        df1, df2 = store[ColumnType.numerical]
        columns = store.column_names(ColumnType.numerical)
        return df1, df2, columns

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return kolmogorov_smirnov_test(part1, part2)

    @staticmethod
    def cumulative_hist_absolute_figure(column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        cumsum1, _, _ = plt.hist(df1[column], bins=bin_edges, align='right', color='cornflowerblue', cumulative=True,
                                 histtype='step')
        cumsum2, _, _ = plt.hist(df2[column], bins=bin_edges, align='right', alpha=0.5, color='seagreen',
                                 cumulative=True, histtype='step')
        distances = abs(cumsum1 - cumsum2)
        max_idx = list(distances).index(max(distances))
        max_d = max(distances)
        plt.plot([bin_edges[max_idx], bin_edges[max_idx]], [cumsum1[max_idx], cumsum2[max_idx]],
                 color='black', linewidth=1, linestyle='--')
        plt.legend(['maximal distance = ' + str(max_d), column + ' 1', column + ' 2'], fontsize='x-small')
        plt.title(column + ' (Cumulative Distribution)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('number of rows', fontsize='medium')
        plt.show()

    @staticmethod
    def cumulative_hist_figure(column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        cumsum1, cumsum2 = plot_cumulative_step_ratio_histogram(df1[column], df2[column], bin_edges)
        distances = abs(cumsum1 - cumsum2)
        max_idx = list(distances).index(max(distances))
        max_d = max(distances)
        plt.plot([bin_edges[max_idx], bin_edges[max_idx]], [cumsum1[max_idx], cumsum2[max_idx]],
                 color='black', linewidth=1, linestyle='--')
        plt.legend([column + ' 1', column + ' 2', 'maximal distance = ' + str(max_d)], fontsize='x-small')
        plt.title(column + ' (Cumulative Distribution)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('ratio of rows', fontsize='medium')
        plt.show()

    @staticmethod
    def overlayed_hist_absolute_figure(column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        plt.hist(df1[column], bins=bin_edges, color='cornflowerblue')
        plt.hist(df2[column], bins=bin_edges, alpha=0.5, color='seagreen')
        plt.legend([column + ' 1', column + ' 2'], fontsize='x-small')
        plt.title(column + ' (Histogram)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('number of rows', fontsize='medium')
        plt.show()

    @staticmethod
    def overlayed_hist_figure(column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        plot_ratio_histogram(df1[column], df2[column], bin_edges)
        plt.legend([column + ' 1', column + ' 2'], fontsize='x-small')
        plt.title(column + ' (Histogram)', fontsize='x-large')
        plt.xlabel('column value', fontsize='medium')
        plt.ylabel('ratio of rows', fontsize='medium')
        plt.show()

    @staticmethod
    def column_figure(column, df1, df2):
        NumericalStatisticalCheck.cumulative_hist_figure(column, df1, df2)
        NumericalStatisticalCheck.overlayed_hist_figure(column, df1, df2)
