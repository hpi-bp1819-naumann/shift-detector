import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.precalculations.binning_precalculation import BinningPrecalculation
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.precalculations.low_cardinality_precalculation import LowCardinalityPrecalculation
from shift_detector.utils import visualization as vis
from shift_detector.utils.column_management import ColumnType


def chi2_test(part1: pd.Series, part2: pd.Series):
    observed = pd.DataFrame.from_dict({'a': part1.value_counts(), 'b': part2.value_counts()})
    observed['a'] = observed['a'].add(1, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(1, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


class CategoricalStatisticalCheck(SimpleStatisticalCheck):

    def check_name(self) -> str:
        return 'Categorical Statistical Check'

    def statistical_test_name(self) -> str:
        return 'Chi^2-Test with Log-Likelihood (G-Test)'

    def data_to_process(self, store):
        cat1, cat2, cat_col = store[LowCardinalityPrecalculation()]
        num1, num2, num_col = store[BinningPrecalculation(bins=20)]
        #lda_text = store[LdaEmbedding(store.column_names(ColumnType.text))]
        df1 = pd.concat([cat1, num1], axis='columns')
        df2 = pd.concat([cat2, num2], axis='columns')
        columns = set(cat_col).union(set(num_col))
        return df1, df2, columns

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return chi2_test(part1, part2)

    @staticmethod
    def paired_total_ratios_plot(figure, axes, column, df1, df2, top_k=15):
        axes = vis.plot_categorical_horizontal_ratio_histogram(axes, (df1[column], df2[column]), top_k)
        axes.invert_yaxis()  # to match order of legend
        column_name = str(column)
        axes.set_title('Column: {}'.format(column_name), fontsize='x-large')
        axes.set_xlabel('value ratio', fontsize='medium')
        axes.set_ylabel('column value', fontsize='medium')
        figure.add_subplot(axes)

    def number_of_columns_of_plots(self) -> int:
        return 2

    @staticmethod
    def column_plot(figure, tile, column, df1, df2):
        CategoricalStatisticalCheck.paired_total_ratios_plot(figure=figure,
                                                             axes=plt.Subplot(figure, tile),
                                                             column=column, df1=df1, df2=df2)
