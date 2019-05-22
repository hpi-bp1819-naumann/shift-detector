import pandas as pd
from scipy import stats

from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport


def chi2_test(part1: pd.Series, part2: pd.Series):
    observed = pd.DataFrame.from_dict({'a': part1.value_counts(), 'b': part2.value_counts()})
    observed['a'] = observed['a'].add(1, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(1, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


class CategoricalStatisticalCheck(Check):

    def run(self, store) -> StatisticalReport:
        pvalues = pd.DataFrame(index=['pvalue'])
        for df1, df2 in [store[key] for key in [ColumnType.categorical]]:
            sample_size = min(len(df1), len(df2))
            for column in store.columns:
                p = chi2_test(df1.sample(sample_size)[column], df2.sample(sample_size)[column])
                pvalues[column] = [p]
        return StatisticalReport(pvalues)
