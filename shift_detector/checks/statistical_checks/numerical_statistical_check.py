import pandas as pd
from scipy import stats

from shift_detector.checks.statistical_checks.statistical_check import SimpleStatisticalCheck
from shift_detector.utils.column_management import ColumnType


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(SimpleStatisticalCheck):

    def statistical_test_name(self) -> str:
        return 'Kolmogorov-Smirnov-Two-Sample-Test'

    def data_to_process(self, store):
        df1, df2 = store[ColumnType.numerical]
        columns = store.column_names(ColumnType.numerical)
        return df1, df2, columns

    def statistical_test(self, part1: pd.Series, part2: pd.Series) -> float:
        return kolmogorov_smirnov_test(part1, part2)
