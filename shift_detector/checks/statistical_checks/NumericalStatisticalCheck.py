import pandas as pd
from datawig.utils import random_split
from scipy import stats

from shift_detector.checks.Check import Check, Report
from shift_detector.Utils import ColumnType
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.Default import Default


class StatisticalNumericalCheck(Check):

    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return 'StatisticalNumericalCheck'

    @staticmethod
    def report_class():
        return StatisticalReport

    def needed_preprocessing(self) -> dict:
        return {
            ColumnType.numeric: Default()
            # TODO: ColumnType.text: Embedding()
        }

    @staticmethod
    def kolmogorov_smirnov_test(column, part1, part2):
        ks_test_result = stats.ks_2samp(part1[column], part2[column])
        return ks_test_result.pvalue

    def run(self, columns=[]) -> pd.DataFrame:
        num_stats = pd.DataFrame()
        for df1, df2 in [self.data[key] for key in self.needed_preprocessing().keys()]:
            df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
            df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
            sample_size = min(len(df1), len(df2))
            for column in df1.columns:
                base1_p = self.kolmogorov_smirnov_test(column, df1_part1, df1_part2)
                base2_p = self.kolmogorov_smirnov_test(column, df2_part1, df2_part2)
                p = self.kolmogorov_smirnov_test(column, df1.sample(sample_size), df2.sample(sample_size))
                num_stats[column] = [base1_p, base2_p, p]
        num_stats.index = ['base1', 'base2', 'test']
        return num_stats
