import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.utils.column_management import ColumnType


def chi2_test(part1: pd.Series, part2: pd.Series):
    observed = pd.DataFrame.from_dict({'a': part1.value_counts(), 'b': part2.value_counts()})
    observed['a'] = observed['a'].add(1, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(1, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


class CategoricalStatisticalCheck(SimpleStatisticalCheck):

    def statistical_test_name(self) -> str:
        return 'Chi^2-Test with Log-Likelihood (G-Test)'

    def store_keys(self):
        return [ColumnType.all_categorical]

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return chi2_test(part1, part2)

    def column_figure(self, column, df1, df2):
        value_counts = pd.concat([df1[column].value_counts().head(50), df2[column].value_counts().head(50)],
                                 axis=1, sort=False)
        value_counts.columns = [column + ' 1', column + ' 2']
        value_counts = value_counts.fillna(0).apply(axis='columns',
                                                    func=lambda row: pd.Series([row.iloc[0] / sum(row),
                                                                                row.iloc[1] / sum(row)],
                                                                               index=[column + ' 1', column + ' 2']))
        axes = value_counts.plot(kind='bar', fontsize='medium', stacked=True)
        axes.legend(fontsize='x-small')
        axes.set_title(column, fontsize='x-large')
        axes.set_xlabel('column value', fontsize='medium')
        axes.set_ylabel('ratio of first data set', fontsize='medium')
        plt.axhline(y=0.5, linestyle='--', linewidth=2, color='black')
        plt.text(x=0.5, y=0.54, s='evenly distributed', fontsize='medium',
                 horizontalalignment='center', verticalalignment='center')
        plt.show()
