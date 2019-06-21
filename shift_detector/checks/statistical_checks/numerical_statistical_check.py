import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import gridspec
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.utils.column_management import ColumnType
import shift_detector.utils.visualization as vis


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
    def cumulative_hist_plot(figure, axes, column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        cumsum1, cumsum2 = vis.plot_cumulative_step_ratio_histogram(axes, (df1[column], df2[column]), bin_edges)
        distances = abs(cumsum1 - cumsum2)
        max_idx = list(distances).index(max(distances))
        max_d = max(distances)
        axes.plot([bin_edges[max_idx], bin_edges[max_idx]], [cumsum1[max_idx], cumsum2[max_idx]],
                  color='black', linewidth=1, linestyle='--')
        column_name = column if isinstance(column, str) else '_'.join(column)
        axes.legend([column_name + ' 1', column_name + ' 2', 'maximal distance = ' + str(max_d)], fontsize='x-small')
        axes.set_title('Column: ' + column_name + ' (Cumulative Distribution)')
        axes.set_xlabel('column value', fontsize='medium')
        axes.set_ylabel('ratio of rows', fontsize='medium')
        figure.add_subplot(axes)

    @staticmethod
    def overlayed_hist_plot(figure, axes, column, df1, df2, bins=40):
        _, bin_edges = np.histogram(pd.concat([df1[column], df2[column]]), bins=bins)
        vis.plot_binned_ratio_histogram(axes, (df1[column], df2[column]), bin_edges)
        column_name = column if isinstance(column, str) else '_'.join(column)
        axes.legend([column_name + ' 1', column_name + ' 2'], fontsize='x-small')
        axes.set_title('Column: ' + column_name + ' (Histogram)')
        axes.set_xlabel('column value', fontsize='medium')
        axes.set_ylabel('ratio of rows', fontsize='medium')
        figure.add_subplot(axes)

    def number_of_columns_of_plots(self) -> int:
        return 1

    @staticmethod
    def column_plot(figure, tile, column, df1, df2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=tile, wspace=0.2)
        NumericalStatisticalCheck.cumulative_hist_plot(figure=figure,
                                                       axes=plt.Subplot(figure, inner_grid[0]),
                                                       column=column, df1=df1, df2=df2)
        NumericalStatisticalCheck.overlayed_hist_plot(figure=figure,
                                                      axes=plt.Subplot(figure, inner_grid[1]),
                                                      column=column, df1=df1, df2=df2)
