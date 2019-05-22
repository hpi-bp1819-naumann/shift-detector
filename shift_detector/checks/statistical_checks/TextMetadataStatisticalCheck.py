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
        self.result = check_result

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
        self.metadata_preprocessor = TextMetadata(text_metadata_types)

    def run(self, store) -> StatisticalReport:
        df1, df2 = store[self.metadata_preprocessor]
        df1_part1, df1_part2 = random_split(df1, [0.5, 0.5])
        df2_part1, df2_part2 = random_split(df2, [0.5, 0.5])
        sample_size = min(len(df1), len(df2))
        categorical = self.metadata_preprocessor.selected_categorical_types()
        numerical = self.metadata_preprocessor.selected_numerical_types()
        col_index = pd.MultiIndex.from_product([store.columns, self.metadata_preprocessor.text_metadata_types], names=['column', 'metadata'])
        text_meta_stats = pd.DataFrame(columns=col_index, index=['base1', 'base2', 'test'])
        for column in store.columns:
            for metadata_type in categorical:
                base1_p = chi2_test(df1_part1[(column, metadata_type)], df1_part2[(column, metadata_type)])
                base2_p = chi2_test(df2_part1[(column, metadata_type)], df2_part2[(column, metadata_type)])
                p = chi2_test(df1.sample(sample_size)[(column, metadata_type)], df2.sample(sample_size)[(column, metadata_type)])
                text_meta_stats[(column, metadata_type)] = [base1_p, base2_p, p]
            for metadata_type in numerical:
                base1_p = kolmogorov_smirnov_test(df1_part1[(column, metadata_type)], df1_part2[(column, metadata_type)])
                base2_p = kolmogorov_smirnov_test(df2_part1[(column, metadata_type)], df2_part2[(column, metadata_type)])
                p = kolmogorov_smirnov_test(df1.sample(sample_size)[(column, metadata_type)],
                              df2.sample(sample_size)[(column, metadata_type)])
                text_meta_stats[(column, metadata_type)] = [base1_p, base2_p, p]
        return StatisticalReport(text_meta_stats)
