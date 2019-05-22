import pandas as pd
from scipy import stats

from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(Check):

    def run(self, store) -> StatisticalReport:
        pvalues = pd.DataFrame(index=['pvalue'])
        for df1, df2 in [store[key] for key in [ColumnType.numeric]]:
            sample_size = min(len(df1), len(df2))
            for column in store.columns:
                p = kolmogorov_smirnov_test(df1.sample(sample_size)[column], df2.sample(sample_size)[column])
                pvalues[column] = [p]
        return StatisticalReport(pvalues)
