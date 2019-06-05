import pandas as pd

from shift_detector.checks.Check import Report
from shift_detector.checks.statistical_checks.CategoricalStatisticalCheck import chi2_test
from shift_detector.checks.statistical_checks.NumericalStatisticalCheck import kolmogorov_smirnov_test
from shift_detector.checks.statistical_checks.StatisticalCheck import StatisticalCheck
from shift_detector.precalculations.TextMetadata import TextMetadata
from shift_detector.utils.ColumnManagement import ColumnType


class TextMetadataStatisticalCheck(StatisticalCheck):

    def __init__(self, text_metadata_types=None, language='en', infer_language=False, significance=0.01,
                 use_sampling=False, sampling_seed=None):
        super().__init__(significance, use_sampling, sampling_seed)
        self.metadata_precalculation = TextMetadata(text_metadata_types, language=language,
                                                    infer_language=infer_language)

    def significant_columns(self, pvalues):
        return set(column for column in pvalues.columns.levels[0]
                   if len(super(type(self), self).significant_columns(pvalues[column])) > 0)

    def significant_metadata(self, mdtype_pvalues):
        return super().significant_columns(mdtype_pvalues)

    def explain(self, pvalues):
        explanation = {}
        for column in self.significant_columns(pvalues):
            explanation[column] = 'Text metadata metrics on column \'{column}\' are unlikely to be equally ' \
                                  'distributed.\n{significant_metadata}'\
                                  .format(
                                    column=column,
                                    significant_metadata='\n'.join(self.significant_metadata(pvalues[column]))
                                  )
        return explanation

    def run(self, store) -> Report:
        df1, df2 = store[self.metadata_precalculation]
        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.use_sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.use_sampling else df2
        pvalues = pd.DataFrame(columns=df1.columns, index=['pvalue'])
        for column in df1.columns.levels[0]:
            for mdtype in self.metadata_precalculation.text_metadata_types:
                if mdtype.metadata_column_type() == ColumnType.categorical:
                    p = chi2_test(part1[(column, mdtype.metadata_name())], part2[(column, mdtype.metadata_name())])
                elif mdtype.metadata_column_type() == ColumnType.numerical:
                    p = kolmogorov_smirnov_test(part1[(column, mdtype.metadata_name())],
                                                part2[(column, mdtype.metadata_name())])
                else:
                    raise ValueError('Return column type {type} of {metadata} is unknown. '
                                     'Should be numerical or categorical.'
                                     .format(type=mdtype.metadata_column_type(), metadata=mdtype))
                pvalues[(column, mdtype.metadata_name())] = [p]
        return Report(examined_columns=list(df1.columns.levels[0]),
                      shifted_columns=self.significant_columns(pvalues),
                      explanation=self.explain(pvalues),
                      information={'test_results': pvalues})
