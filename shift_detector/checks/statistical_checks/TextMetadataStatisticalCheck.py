import pandas as pd

from shift_detector.checks.Check import Check
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.TextMetadata import TextMetadata


class TextMetadataStatisticalCheck(Check):

    def __init__(self, text_metadata_types=None):
        self.metadata_preprocessor = TextMetadata(text_metadata_types)

    def run(self, store) -> StatisticalReport:
        df1, df2 = store[self.metadata_preprocessor]
        sample_size = min(len(df1), len(df2))
        categorical = self.metadata_preprocessor.selected_categorical_types()
        numerical = self.metadata_preprocessor.selected_numerical_types()
        col_index = pd.MultiIndex.from_product([store.columns, self.metadata_preprocessor.text_metadata_types],
                                               names=['column', 'metadata'])
        text_meta_stats = pd.DataFrame(columns=col_index, index=['pvalue'])
        for column in store.columns:
            for metadata_type in categorical:
                p = chi2_test(df1.sample(sample_size)[(column, metadata_type)],
                              df2.sample(sample_size)[(column, metadata_type)])
                text_meta_stats[(column, metadata_type)] = [p]
            for metadata_type in numerical:
                p = kolmogorov_smirnov_test(df1.sample(sample_size)[(column, metadata_type)],
                                            df2.sample(sample_size)[(column, metadata_type)])
                text_meta_stats[(column, metadata_type)] = [p]
        return StatisticalReport(text_meta_stats)
