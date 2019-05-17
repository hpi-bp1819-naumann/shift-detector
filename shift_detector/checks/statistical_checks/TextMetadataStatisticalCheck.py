import pandas as pd
from datawig.utils import random_split

from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check, Report
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.TextMetadata import TextMetadata


class TextMetadataStatisticalReport(Report):

    def __init__(self, check_result, significance=0.01):
        super().__init__(check_result)
        self.significance = significance

    def is_significant(self, base1_p: float, base2_p: float, p: float) -> bool:
        return (base1_p > self.significance
                and base2_p > self.significance
                and p <= self.significance)

    def significant_columns(self):
        columns = set()
        for column in self.result.columns.levels[0]:
            columns_metadata = pd.DataFrame.from_records(self.result[column], index=['base1', 'base2', 'test'])
            if any(self.is_significant(columns_metadata[md].loc['base1'],
                                       columns_metadata[md].loc['base2'],
                                       columns_metadata[md].loc['test']) for md in columns_metadata.columns):
                columns.add(column)
        return columns

    def significant_text_metadata(self, column):
        columns_metadata = pd.DataFrame.from_records(self.result[column], index=['base1', 'base2', 'test'])
        return set(md for md in columns_metadata.columns if self.is_significant(columns_metadata[md].loc['base1'],
                                                                                columns_metadata[md].loc['base2'],
                                                                                columns_metadata[md].loc['test']))

    def print_report(self):
        print('Columns with probability for equal distribution below significance level ', self.significance, ': ')
        print(self.significant_columns())


class TextMetadataStatisticalCheck(Check):

    def __init__(self, text_metadata_types=None):
        super().__init__()
        self.metadata_preprocessor = TextMetadata(text_metadata_types)

    @staticmethod
    def name() -> str:
        return 'TextMetadataStatisticalCheck'

    @staticmethod
    def report_class():
        return StatisticalReport

    def needed_preprocessing(self) -> dict:
        return {
            ColumnType.text: self.metadata_preprocessor
        }

    def run(self, columns=[]) -> pd.DataFrame:
        df1, df2 = self.data[ColumnType.text]
        df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
        df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
        sample_size = min(len(df1), len(df2))
        categorical = self.metadata_preprocessor.categorical_columns()
        numerical = self.metadata_preprocessor.numerical_columns()
        index = pd.MultiIndex.from_product([df1.columns, categorical.union(numerical)], names=['column', 'metadata'])
        text_meta_stats = pd.DataFrame(columns=index)
        for column in categorical:
            print('cross val 1')
            base1_ps = chi2_test(df1_part1, df1_part2)
            print('cross val 2')
            base2_ps = chi2_test(df2_part1, df2_part2)
            print('real test')
            ps = chi2_test(df1.sample(sample_size), df2.sample(sample_size))
            for metadata_type in categorical:
                text_meta_stats[(column, metadata_type)] = [base1_ps[metadata_type], base2_ps[metadata_type],
                                                            ps[metadata_type]]
        for column in numerical:
            print('cross val 1')
            base1_ps = kolmogorov_smirnov_test(df1_part1, df1_part2)
            print('cross val 2')
            base2_ps = kolmogorov_smirnov_test(df2_part1, df2_part2)
            print('real test')
            ps = kolmogorov_smirnov_test(df1.sample(sample_size), df2.sample(sample_size))
            for metadata_type in numerical:
                text_meta_stats[(column, metadata_type)] = [base1_ps[metadata_type], base2_ps[metadata_type],
                                                            ps[metadata_type]]
        return text_meta_stats
