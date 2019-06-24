import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.precalculations.low_cardinality_precalculation import LowCardinalityPrecalculation


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
        df1, df2, columns = store[LowCardinalityPrecalculation()]
        return df1, df2, columns

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return chi2_test(part1, part2)

    @staticmethod
    def paired_total_ratios_figure(column, df1, df2, top_k=50):
        value_counts = pd.concat([df1[column].value_counts().head(top_k), df2[column].value_counts().head(top_k)],
                                 axis=1).sort_index()
        value_ratios = value_counts.fillna(0).apply(axis='columns',
                                                    func=lambda row: pd.Series([row.iloc[0] / len(df1[column]),
                                                                                row.iloc[1] / len(df2[column])],
                                                                               index=[str(column) + ' 1',
                                                                                      str(column) + ' 2']))
        axes = value_ratios.plot(kind='barh', fontsize='medium')
        axes.invert_yaxis()  # to match order of legend
        column_name = column if isinstance(column, str) else '_'.join(column)
        axes.set_title('Column: ' + column_name, fontsize='x-large')
        axes.set_xlabel('value ratio', fontsize='medium')
        axes.set_ylabel('column value', fontsize='medium')
        plt.show()

    @staticmethod
    def column_figure(column, df1, df2):
        CategoricalStatisticalCheck.paired_total_ratios_figure(column, df1, df2)
