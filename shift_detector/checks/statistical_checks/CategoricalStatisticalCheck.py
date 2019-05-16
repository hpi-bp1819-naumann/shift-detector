import pandas as pd
from datawig.utils import random_split
from scipy import stats

from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check, Report
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.Default import Default


class StatisticalCategoricalCheck(Check):

    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return 'StatisticalCategoricalCheck'

    @staticmethod
    def report_class():
        return StatisticalReport

    def needed_preprocessing(self) -> dict:
        return {
            ColumnType.categorical: Default()
            # TODO: ColumnType.numeric: Binning()
            # TODO: ColumnType.text: Clustering()
        }

    @staticmethod
    def chi2_test(column, part1, part2):
        observed = pd.DataFrame.from_dict({'a': part1[column].value_counts(), 'b': part2[column].value_counts()})
        observed['a'] = observed['a'].add(5, fill_value=0)  # rule of succession
        observed['b'] = observed['b'].add(5, fill_value=0)
        chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
        return p

    def run(self, columns=[]) -> pd.DataFrame:
        cat_stats = pd.DataFrame()
        for df1, df2 in [self.data[key] for key in self.needed_preprocessing().keys()]:
            df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
            df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
            sample_size = min(len(df1), len(df2))
            for column in df1.columns:
                base1_p = self.chi2_test(column, df1_part1, df1_part2)
                base2_p = self.chi2_test(column, df2_part1, df2_part2)
                p = self.chi2_test(column, df1.sample(sample_size), df2.sample(sample_size))
                cat_stats[column] = [base1_p, base2_p, p]
        cat_stats.index = ['base1', 'base2', 'test']
        return cat_stats
