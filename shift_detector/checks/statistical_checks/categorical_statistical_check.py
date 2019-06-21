import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.precalculations.low_cardinality_precalculation import LowCardinalityPrecalculation
from shift_detector.utils import visualization as vis


def chi2_test(part1: pd.Series, part2: pd.Series):
    observed = pd.DataFrame.from_dict({'a': part1.value_counts(), 'b': part2.value_counts()})
    observed['a'] = observed['a'].add(1, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(1, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


class CategoricalStatisticalCheck(SimpleStatisticalCheck):

    def statistical_test_name(self) -> str:
        return 'Chi^2-Test with Log-Likelihood (G-Test)'

    def data_to_process(self, store):
        df1, df2, columns = store[LowCardinalityPrecalculation()]
        return df1, df2, columns

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return chi2_test(part1, part2)

    @staticmethod
    def paired_total_ratios_plot(figure, axes, column, df1, df2, top_k=50):
        axes = vis.plot_categorical_horizontal_ratio_histogram(axes, (df1[column], df2[column]), top_k)
        axes.invert_yaxis()  # to match order of legend
        column_name = column if isinstance(column, str) else '_'.join(column)
        axes.set_title('Column: ' + column_name, fontsize='x-large')
        axes.set_xlabel('value ratio', fontsize='medium')
        axes.set_ylabel('column value', fontsize='medium')
        figure.add_subplot(axes)

    def number_of_columns_of_plots(self) -> int:
        return 2

    @staticmethod
    def column_plot(figure, tile, column, df1, df2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=tile, wspace=0.2)
        CategoricalStatisticalCheck.paired_total_ratios_plot(figure=figure,
                                                             axes=plt.Subplot(figure, inner_grid[0]),
                                                             column=column, df1=df1, df2=df2)
