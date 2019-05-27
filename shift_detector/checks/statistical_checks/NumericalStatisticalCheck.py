import pandas as pd
from scipy import stats

from shift_detector.checks.statistical_checks.StatisticalCheck import SimpleStatisticalCheck
from shift_detector.utils.ColumnManagement import ColumnType


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(SimpleStatisticalCheck):

    def store_keys(self):
        return [ColumnType.numerical]

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return kolmogorov_smirnov_test(part1, part2)
