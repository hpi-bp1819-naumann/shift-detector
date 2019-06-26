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

    def __init__(self, significance=0.01, use_sampling=False, sampling_seed=0, use_binning=False, use_embedding=False):
        self.use_binning = use_binning
        self.use_embedding = use_embedding
        super().__init__(significance, use_sampling, sampling_seed)

    def check_name(self) -> str:
        return 'Categorical Statistical Check'

    def statistical_test_name(self) -> str:
        return 'Chi^2-Test with Log-Likelihood (G-Test)'

    def data_to_process(self, store):
        if not self.use_binning:
            cat1, cat2, cat_col = store[LowCardinalityPrecalculation()]
        else:
            cat1, cat2 = store[ColumnType.categorical]
            cat_col = store.column_names(ColumnType.categorical)
        df1 = cat1
        df2 = cat2
        columns = set(cat_col)
        if self.use_binning:
            num1, num2, num_col = store[BinningPrecalculation(bins=20)]
            df1 = pd.concat([df1, num1], axis='columns')
            df2 = pd.concat([df2, num2], axis='columns')
            columns = columns.union(set(num_col))
        if self.use_embedding:
            embedding = LdaEmbedding(store.column_names(ColumnType.text))
            lda_text = store[embedding]
            topic_num_to_label = dict(zip(range(embedding.n_topics),
                                          ['topic_{}'.format(i) for i in range(embedding.n_topics)]))
            lda1 = lda_text[0].replace(topic_num_to_label)
            lda2 = lda_text[1].replace(topic_num_to_label)
            df1 = pd.concat([df1, lda1], axis='columns')
            df2 = pd.concat([df2, lda2], axis='columns')
            columns = columns.union(set(lda1.columns))
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
        return 1

    @staticmethod
    def column_plot(figure, tile, column, df1, df2):
        CategoricalStatisticalCheck.paired_total_ratios_plot(figure=figure,
                                                             axes=plt.Subplot(figure, tile),
                                                             column=column, df1=df1, df2=df2)
