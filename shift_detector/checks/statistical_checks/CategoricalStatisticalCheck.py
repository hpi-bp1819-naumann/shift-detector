import pandas as pd
from scipy import stats

from shift_detector.checks.statistical_checks.StatisticalCheck import SimpleStatisticalCheck
from shift_detector.utils.ColumnManagement import ColumnType


def chi2_test(part1: pd.Series, part2: pd.Series):
    observed = pd.DataFrame.from_dict({'a': part1.value_counts(), 'b': part2.value_counts()})
    observed['a'] = observed['a'].add(1, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(1, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


class CategoricalStatisticalCheck(SimpleStatisticalCheck):

    def store_keys(self):
        return [ColumnType.all_categorical]

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return chi2_test(part1, part2)
