import pandas as pd
from datawig.utils import random_split
from scipy import stats

from shift_detector.checks.Check import Check, Report
from shift_detector.Utils import ColumnType
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.Default import Default


def kolmogorov_smirnov_test(part1: pd.Series, part2: pd.Series):
    ks_test_result = stats.ks_2samp(part1, part2)
    return ks_test_result.pvalue


class NumericalStatisticalCheck(Check):

    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return 'NumericalStatisticalCheck'

    @staticmethod
    def report_class():
        return StatisticalReport

    def needed_preprocessing(self) -> dict:
        return {
            ColumnType.numeric: Default()
            # TODO: ColumnType.text: Embedding()
        }

    def run(self, columns=[]) -> pd.DataFrame:
        num_stats = pd.DataFrame()
        for df1, df2 in [self.data[key] for key in self.needed_preprocessing().keys()]:
            df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
            df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
            sample_size = min(len(df1), len(df2))
            for column in df1.columns:
                base1_p = kolmogorov_smirnov_test(df1_part1[column], df1_part2[column])
                base2_p = kolmogorov_smirnov_test(df2_part1[column], df2_part2[column])
                p = kolmogorov_smirnov_test(df1.sample(sample_size)[column], df2.sample(sample_size)[column])
                num_stats[column] = [base1_p, base2_p, p]
        num_stats.index = ['base1', 'base2', 'test']
        return num_stats
