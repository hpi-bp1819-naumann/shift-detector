import pandas as pd
from datawig.utils import random_split
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
        cat_stats = pd.DataFrame()
        for df1, df2 in [store[key] for key in [ColumnType.categorical]]:
            df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
            df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
            sample_size = min(len(df1), len(df2))
            for column in df1.columns:
                base1_p = chi2_test(df1_part1[column], df1_part2[column])
                base2_p = chi2_test(df2_part1[column], df2_part2[column])
                p = chi2_test(df1.sample(sample_size)[column], df2.sample(sample_size)[column])
                cat_stats[column] = [base1_p, base2_p, p]
        cat_stats.index = ['base1', 'base2', 'test']
        return StatisticalReport(cat_stats)
