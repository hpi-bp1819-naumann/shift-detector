import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from shift_detector.checks.check import Report
from shift_detector.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from shift_detector.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck
from shift_detector.checks.statistical_checks.statistical_check import StatisticalCheck, StatisticalReport
from shift_detector.precalculations.text_metadata import TextMetadata
from shift_detector.utils.column_management import ColumnType
from shift_detector.utils.errors import UnknownMetadataReturnColumnTypeError
from shift_detector.utils.visualization import PLOT_GRID_WIDTH, PLOT_ROW_HEIGHT


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
        return set(mdtype for mdtype in self.metadata_precalculation.text_metadata_types
                   if mdtype.metadata_name() in super(type(self), self).significant_columns(mdtype_pvalues))

    def significant_metadata_names(self, mdtype_pvalues):
        return [mdtype.metadata_name() for mdtype in self.significant_metadata(mdtype_pvalues)]

    def explanation_header(self, numerical_test_name, categorical_test_name, any_significant):
        header = 'Statistical tests performed:\n' + \
                 '\t- numerical metadata: {}\n'.format(numerical_test_name) + \
                 '\t- categorical metadata: {}\n'.format(categorical_test_name) + \
                 'Significance level: {}\n'.format(str(self.significance))
        if any_significant:
            header += '\nSome text metadata metrics on the following columns are unlikely to be equally distributed.\n'
        return header

    def explain(self, pvalues):
        explanation = {}
        for column in self.significant_columns(pvalues):
            explanation[column] = 'Significant metadata:\n\t\t- {significant_metadata}'.format(
                significant_metadata='\n\t\t- '.join(self.significant_metadata_names(pvalues[column]))
            )
        return explanation

    @staticmethod
    def metadata_plot(figure, tile, column, mdtype, df1, df2):
        col_mdtype_tuple = (column, mdtype.metadata_name())
        if mdtype.metadata_return_type() == ColumnType.categorical:
            CategoricalStatisticalCheck.column_plot(figure, tile, col_mdtype_tuple, df1, df2)
        elif mdtype.metadata_return_type() == ColumnType.numerical:
            NumericalStatisticalCheck.column_plot(figure, tile, col_mdtype_tuple, df1, df2)
        else:
            raise UnknownMetadataReturnColumnTypeError(mdtype)

    @staticmethod
    def plot_all_metadata(plot_functions):
        rows = len(plot_functions)
        cols = 1
        fig = plt.figure(figsize=(PLOT_GRID_WIDTH, PLOT_ROW_HEIGHT * rows), tight_layout=True)
        grid = gridspec.GridSpec(rows, cols)
        for plot_function, tile in zip(plot_functions, grid):
            plot_function(fig, tile)
        plt.show()

    def plot_functions(self, significant_columns, pvalues, df1, df2):
        plot_functions = []
        for column in sorted(significant_columns):
            for mdtype in sorted(self.significant_metadata(pvalues[column])):
                plot_functions.append(lambda figure, tile, col=column, meta=mdtype:
                                      self.metadata_plot(figure, tile, col, meta, df1, df2))
        return plot_functions

    def metadata_figure(self, pvalues, df1, df2):
        significant_columns = self.significant_columns(pvalues)
        if not significant_columns:
            return []
        return [lambda plots=tuple(self.plot_functions(significant_columns, pvalues, df1, df2)):
                self.plot_all_metadata(plots)]

    def run(self, store) -> Report:
        df1, df2 = store[self.metadata_precalculation]
        sample_size = min(len(df1), len(df2))
        part1 = df1.sample(sample_size, random_state=self.seed) if self.use_sampling else df1
        part2 = df2.sample(sample_size, random_state=self.seed) if self.use_sampling else df2
        categorical_check = CategoricalStatisticalCheck()
        numerical_check = NumericalStatisticalCheck()
        pvalues = pd.DataFrame(columns=df1.columns, index=['pvalue'])
        for column in df1.columns.levels[0]:
            for mdtype in self.metadata_precalculation.text_metadata_types:
                if mdtype.metadata_return_type() == ColumnType.categorical:
                    p = categorical_check.statistical_test(part1[(column, mdtype.metadata_name())],
                                                           part2[(column, mdtype.metadata_name())])
                elif mdtype.metadata_return_type() == ColumnType.numerical:
                    p = numerical_check.statistical_test(part1[(column, mdtype.metadata_name())],
                                                         part2[(column, mdtype.metadata_name())])
                else:
                    raise UnknownMetadataReturnColumnTypeError(mdtype)
                pvalues[(column, mdtype.metadata_name())] = [p]
        significant_columns = self.significant_columns(pvalues)
        return StatisticalReport("Text Metadata Check",
                                 examined_columns=list(df1.columns.levels[0]),
                                 shifted_columns=significant_columns,
                                 explanation=self.explain(pvalues),
                                 explanation_header=self.explanation_header(numerical_check.statistical_test_name(),
                                                                            categorical_check.statistical_test_name(),
                                                                            any_significant=len(significant_columns) > 0
                                                                            ),
                                 information={'test_results': pvalues},
                                 figures=self.metadata_figure(pvalues, part1, part2))
