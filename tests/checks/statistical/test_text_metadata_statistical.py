import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.util.testing import assert_frame_equal

import tests.test_data as td
from shift_detector.checks.statistical_checks import numerical_statistical_check, categorical_statistical_check
from shift_detector.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from shift_detector.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck
from shift_detector.checks.statistical_checks.text_metadata_statistical_check import TextMetadataStatisticalCheck
from shift_detector.detector import Detector
from shift_detector.precalculations.store import Store
from shift_detector.precalculations.text_metadata import NumCharsMetadata, NumWordsMetadata, \
    DistinctWordsRatioMetadata, LanguagePerParagraph, UnknownWordRatioMetadata, StopwordRatioMetadata, LanguageMetadata
from shift_detector.utils.visualization import PlotData


class TestTextMetadataStatisticalCheck(unittest.TestCase):

    def setUp(self):
        self.poems = td.poems
        self.phrases = td.phrases

    def test_significant_metadata(self):
        pvalues = pd.DataFrame([[0.001, 0.2]], columns=['num_chars', 'distinct_words_ratio'], index=['pvalue'])
        result = TextMetadataStatisticalCheck(significance=0.01).significant_metadata_names(pvalues)
        self.assertIn('num_chars', result)
        self.assertNotIn('distinct_words_ratio', result)

    def test_not_significant(self):
        df1 = pd.DataFrame.from_dict({'text': self.poems})
        df2 = pd.DataFrame.from_dict({'text': list(reversed(self.poems))})
        store = Store(df1, df2)
        result = TextMetadataStatisticalCheck().run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(0, len(result.shifted_columns))
        self.assertEqual(0, len(result.explanation))

    def test_significant(self):
        df1 = pd.DataFrame.from_dict({'text': self.poems})
        df2 = pd.DataFrame.from_dict({'text': self.phrases})
        store = Store(df1, df2)
        result = TextMetadataStatisticalCheck([NumCharsMetadata(), NumWordsMetadata(),
                                               DistinctWordsRatioMetadata(), LanguagePerParagraph()]
                                              ).run(store)
        self.assertEqual(1, len(result.examined_columns))
        self.assertEqual(1, len(result.shifted_columns))
        self.assertEqual(1, len(result.explanation))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame.from_dict({'text': ['This is a very important text.',
                                               'It contains information.', 'Brilliant ideas are written down.',
                                               'Read it.', 'You will become a lot smarter.',
                                               'Or you will waste your time.', 'Come on, figure it out!',
                                               'Perhaps it will at least entertain you.', 'Do not be afraid.',
                                               'Be brave!']})
        df2 = pd.DataFrame.from_dict({'text': ['This is a very important text.',
                                               'It contains information.', 'Brilliant ideas are written down.',
                                               'Read it.', 'You will become a lot smarter.',
                                               'Or you will waste your time.', 'Come on, figure it out!',
                                               'Perhaps it will at least entertain you.', 'Do not be afraid.',
                                               'Be brave!']})
        detector = Detector(df1=df1, df2=df2, log_print=False)
        detector.run(TextMetadataStatisticalCheck())
        column_index = pd.MultiIndex.from_product([['text'], ['distinct_words', 'num_chars', 'num_words']],
                                                  names=['column', 'metadata'])
        solution = pd.DataFrame([[1.0, 1.0, 1.0]], columns=column_index, index=['pvalue'])
        self.assertEqual(1, len(detector.check_reports[0].examined_columns))
        self.assertEqual(0, len(detector.check_reports[0].shifted_columns))
        self.assertEqual(0, len(detector.check_reports[0].explanation))
        assert_frame_equal(solution, detector.check_reports[0].information['test_results'])

    def test_language_can_be_set(self):
        check = TextMetadataStatisticalCheck([UnknownWordRatioMetadata(), StopwordRatioMetadata()], language='fr')
        md_with_lang = [mdtype for mdtype in check.metadata_precalculation.text_metadata_types
                        if type(mdtype) in [UnknownWordRatioMetadata, StopwordRatioMetadata]]
        for mdtype in md_with_lang:
            self.assertEqual('fr', mdtype.language)

    def test_infer_language_is_set(self):
        check = TextMetadataStatisticalCheck([UnknownWordRatioMetadata(), StopwordRatioMetadata()], infer_language=True)
        md_with_lang = [mdtype for mdtype in check.metadata_precalculation.text_metadata_types
                        if type(mdtype) in [UnknownWordRatioMetadata, StopwordRatioMetadata]]
        for mdtype in md_with_lang:
            self.assertTrue(mdtype.infer_language)

    def test_figure_function_is_collected(self):
        df1 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        df2 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        metadata_names = ['num_chars', 'num_words']
        cols = pd.MultiIndex.from_product([df1.columns, metadata_names], names=['column', 'metadata'])
        check = TextMetadataStatisticalCheck()
        pvalues = pd.DataFrame(columns=cols, index=['pvalue'])
        for solution, num_sig_metadata in [(1, 2), (1, 1), (0, 0)]:
            p = [0.001] * num_sig_metadata + [0.05] * (2 - num_sig_metadata)
            pvalues[('text', 'num_chars')] = p[0]
            pvalues[('text', 'num_words')] = p[1]
            with self.subTest(solution=solution, pvalues=pvalues):
                result = check.metadata_figure(pvalues=pvalues, df1=df1, df2=df2)
                self.assertEqual(solution, len(result))

    @mock.patch('shift_detector.checks.statistical_checks.text_metadata_statistical_check.plt')
    def test_all_plot_functions_are_called_and_plot_is_shown(self, mock_plt):
        plot_data = [PlotData(MagicMock(), 1), PlotData(MagicMock(), 2), PlotData(MagicMock(), 3)]
        TextMetadataStatisticalCheck.plot_all_metadata(plot_data)
        mock_plt.figure.assert_called_with(figsize=(12.0, 30.0), tight_layout=True)
        for func, rows in plot_data:
            self.assertTrue(func.called)
        mock_plt.show.assert_called_with()

    def test_column_tuples_are_handled_by_numerical_visualization(self):
        columns = ['text']
        metadata_names = ['num_chars']
        cols = pd.MultiIndex.from_product([columns, metadata_names], names=['column', 'metadata'])
        df1 = pd.DataFrame(columns=cols)
        df2 = pd.DataFrame(columns=cols)
        df1[('text', 'num_chars')] = [1, 2, 3]
        df2[('text', 'num_chars')] = [3, 2, 1]
        mock_figure = MagicMock(autospec=Figure)
        mock_axes = MagicMock(autospec=Axes)
        with mock.patch.object(numerical_statistical_check.vis, 'plot_binned_ratio_histogram'):
            NumericalStatisticalCheck.overlayed_hist_plot(mock_figure, mock_axes, ('text', 'num_chars'), df1, df2)
        mock_axes.legend.assert_called_with(["('text', 'num_chars') 1", "('text', 'num_chars') 2"], fontsize='x-small')
        mock_axes.set_title.assert_called_with("Column: ('text', 'num_chars') (Histogram)")
        with mock.patch.object(numerical_statistical_check.vis, 'plot_cumulative_step_ratio_histogram',
                               return_value=(np.array([0]), np.array([0]))):
            NumericalStatisticalCheck.cumulative_hist_plot(mock_figure, mock_axes, ('text', 'num_chars'), df1, df2)
        mock_axes.legend.assert_called_with(["('text', 'num_chars') 1", "('text', 'num_chars') 2", 'maximal distance = 0'],
                                           fontsize='x-small')
        mock_axes.set_title.assert_called_with("Column: ('text', 'num_chars') (Cumulative Distribution)")

    def test_column_tuples_are_handled_by_categorical_visualization(self):
        columns = ['text']
        metadata_names = ['category']
        cols = pd.MultiIndex.from_product([columns, metadata_names], names=['column', 'metadata'])
        df1 = pd.DataFrame(columns=cols)
        df2 = pd.DataFrame(columns=cols)
        df1[('text', 'category')] = ['latin' * 3]
        df2[('text', 'category')] = ['arabic' * 3]
        mock_figure = MagicMock(autospec=Figure)
        mock_axes = MagicMock(autospec=Axes)
        with mock.patch.object(categorical_statistical_check.vis, 'plot_categorical_horizontal_ratio_histogram',
                               return_value=mock_axes):
            CategoricalStatisticalCheck.paired_total_ratios_plot(mock_figure, mock_axes, ('text', 'category'), df1, df2)
        mock_axes.set_title.assert_called_once_with("Column: ('text', 'category')", fontsize='x-large')

    def test_correct_visualization_is_chosen_categorical(self):
        with mock.patch.object(CategoricalStatisticalCheck, 'column_plot') as mock_plot:
            figure = MagicMock(spec=Figure)
            tile = MagicMock()
            TextMetadataStatisticalCheck.metadata_plot(figure, tile, 'text', LanguageMetadata(), None, None)
        self.assertTrue(mock_plot.called)

    def test_correct_visualization_is_chosen_numerical(self):
        with mock.patch.object(NumericalStatisticalCheck, 'column_plot') as mock_plot:
            figure = MagicMock(spec=Figure)
            tile = MagicMock()
            TextMetadataStatisticalCheck.metadata_plot(figure, tile, 'text', NumCharsMetadata(), None, None)
        self.assertTrue(mock_plot.called)

    def test_correct_number_of_plot_data(self):
        df1 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        df2 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        metadata_names = ['num_chars', 'num_words']
        cols = pd.MultiIndex.from_product([df1.columns, metadata_names], names=['column', 'metadata'])
        check = TextMetadataStatisticalCheck()
        pvalues = pd.DataFrame(columns=cols, index=['pvalue'])
        for num_sig_metadata in [2, 1, 0]:
            p = [0.001] * num_sig_metadata + [0.05] * (2 - num_sig_metadata)
            pvalues[('text', 'num_chars')] = p[0]
            pvalues[('text', 'num_words')] = p[1]
            with self.subTest(num_sig_metadata=num_sig_metadata, pvalues=pvalues):
                result = check.plot_data(['text'], pvalues, df1, df2)
                self.assertEqual(num_sig_metadata, len(result))

    def test_column_order_in_report(self):
        df1 = pd.DataFrame.from_dict({'text': self.poems, 'abc': self.poems})
        df2 = pd.DataFrame.from_dict({'text': self.phrases, 'abc': self.phrases})
        store = Store(df1, df2)
        result = TextMetadataStatisticalCheck([NumCharsMetadata()]).run(store)
        self.assertEqual('abc', result.examined_columns[0])
        self.assertEqual('abc', result.shifted_columns[0])
        self.assertEqual(result.examined_columns, result.shifted_columns)
