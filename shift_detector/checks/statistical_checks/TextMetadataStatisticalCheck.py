import pandas as pd

from shift_detector.checks.Check import Check
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalReport
from shift_detector.preprocessors.TextMetadata import TextMetadata
from shift_detector.utils.ColumnManagement import ColumnType


class TextMetadataStatisticalCheck(Check):

    def __init__(self, text_metadata_types=None, language=None, infer_language=False, sampling=False, sampling_seed=None):
        if not infer_language and language is not None:
            for i in range(len(text_metadata_types)):
                try:
                    text_metadata_types[i].language = language
                except AttributeError:
                    continue  # do nothing for types which do not accept a language as parameter
        elif infer_language:
            for i in range(len(text_metadata_types)):
                try:
                    text_metadata_types[i].infer_language = True
                except AttributeError:
                    continue  # do nothing for types which do not accept a language as parameter
        self.metadata_preprocessor = TextMetadata(text_metadata_types)
        self.sampling = sampling
        self.seed = sampling_seed

    def run(self, store) -> StatisticalReport:
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
        return StatisticalReport(text_meta_stats)
