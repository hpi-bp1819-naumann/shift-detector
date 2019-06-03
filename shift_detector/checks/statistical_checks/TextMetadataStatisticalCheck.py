import pandas as pd

from shift_detector.checks.Check import Check
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalDeprecatedReport
from shift_detector.precalculations.TextMetadata import TextMetadata
from shift_detector.utils.ColumnManagement import ColumnType


class TextMetadataStatisticalCheck(Check):

    def __init__(self, text_metadata_types=None, language='en', infer_language=False, sampling=False, sampling_seed=None):
        self.metadata_preprocessor = TextMetadata(text_metadata_types, language=language, infer_language=infer_language)
        self.sampling = sampling
        self.seed = sampling_seed

    def run(self, store) -> StatisticalDeprecatedReport:
        df1, df2 = store[self.metadata_preprocessor]
        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.sampling else df2
        text_meta_stats = pd.DataFrame(columns=df1.columns, index=['pvalue'])
        for column in df1.columns.get_level_values('column'):
            for mdtype in self.metadata_preprocessor.text_metadata_types:
                if mdtype.metadata_column_type() == ColumnType.categorical:
                    p = chi2_test(part1[(column, mdtype.metadata_name())], part2[(column, mdtype.metadata_name())])
                elif mdtype.metadata_column_type() == ColumnType.numerical:
                    p = kolmogorov_smirnov_test(part1[(column, mdtype.metadata_name())], part2[(column, mdtype.metadata_name())])
                else:
                    raise ValueError('Metadata column type ', mdtype.metadata_column_type(),' of ', mdtype, ' is unknown. Should be numerical or categorical.')
                text_meta_stats[(column, mdtype.metadata_name())] = [p]
        return StatisticalDeprecatedReport(text_meta_stats)
