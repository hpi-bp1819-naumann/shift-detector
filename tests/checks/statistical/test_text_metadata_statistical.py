import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas.util.testing import assert_frame_equal

import tests.test_data as td
from morpheus.checks.statistical_checks import numerical_statistical_check
from morpheus.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from morpheus.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck
from morpheus.checks.statistical_checks.text_metadata_statistical_check import TextMetadataStatisticalCheck
from morpheus.detector import Detector
from morpheus.precalculations.store import Store
from morpheus.precalculations.text_metadata import NumCharsMetadata, NumWordsMetadata, \
    DistinctWordsRatioMetadata, LanguagePerParagraph, UnknownWordRatioMetadata, StopwordRatioMetadata, LanguageMetadata


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
        detector = Detector(df1=df1, df2=df2)
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

    def test_figure_functions_are_collected(self):
        df1 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        df2 = pd.DataFrame.from_dict({'text': ['blub'] * 10})
        metadata_names = ['num_chars', 'num_words']
        cols = pd.MultiIndex.from_product([df1.columns, metadata_names], names=['column', 'metadata'])
        pvalues = pd.DataFrame(columns=cols, index=['pvalue'])
        pvalues[('text', 'num_chars')] = 0.05
        pvalues[('text', 'num_words')] = 0.001
        check = TextMetadataStatisticalCheck()
        result = check.metadata_figures(pvalues=pvalues, df1=df1, df2=df2)
        self.assertEqual(1, len(result))

    @mock.patch('morpheus.checks.statistical_checks.numerical_statistical_check.plt')
    def test_column_tuples_are_handled_by_numerical_visualization(self, mock_plt):
        columns = ['text']
        metadata_names = ['num_chars']
        cols = pd.MultiIndex.from_product([columns, metadata_names], names=['column', 'metadata'])
        df1 = pd.DataFrame(columns=cols)
        df2 = pd.DataFrame(columns=cols)
        df1[('text', 'num_chars')] = [1, 2, 3]
        df2[('text', 'num_chars')] = [3, 2, 1]
        with mock.patch.object(numerical_statistical_check.vis, 'plot_ratio_histogram'):
            NumericalStatisticalCheck.overlayed_hist_figure(('text', 'num_chars'), df1, df2)
        mock_plt.legend.assert_called_with(['text_num_chars 1', 'text_num_chars 2'], fontsize='x-small')
        mock_plt.title.assert_called_with('Column: text_num_chars (Histogram)', fontsize='x-large')
        with mock.patch.object(numerical_statistical_check.vis, 'plot_cumulative_step_ratio_histogram',
                               return_value=(np.array([0]), np.array([0]))):
            NumericalStatisticalCheck.cumulative_hist_figure(('text', 'num_chars'), df1, df2)
        mock_plt.legend.assert_called_with(['text_num_chars 1', 'text_num_chars 2', 'maximal distance = 0'],
                                           fontsize='x-small')
        mock_plt.title.assert_called_with('Column: text_num_chars (Cumulative Distribution)', fontsize='x-large')

    @mock.patch('morpheus.checks.statistical_checks.categorical_statistical_check.plt')
    def test_column_tuples_are_handled_by_categorical_visualization(self, mock_plt):
        columns = ['text']
        metadata_names = ['category']
        cols = pd.MultiIndex.from_product([columns, metadata_names], names=['column', 'metadata'])
        df1 = pd.DataFrame(columns=cols)
        df2 = pd.DataFrame(columns=cols)
        df1[('text', 'category')] = ['latin' * 3]
        df2[('text', 'category')] = ['arabic' * 3]
        mock_axes = MagicMock(spec=Axes)
        with mock.patch.object(pd.DataFrame, 'plot', return_value=mock_axes) as mock_plot:
            CategoricalStatisticalCheck.paired_total_ratios_figure(('text', 'category'), df1, df2)
        mock_axes.set_title.assert_called_once_with('Column: text_category', fontsize='x-large')

    def test_correct_visualization_is_chosen_categorical(self):
        with mock.patch.object(CategoricalStatisticalCheck, 'column_figure') as mock_figure:
            TextMetadataStatisticalCheck.metadata_figure('text', LanguageMetadata(), None, None)
        self.assertTrue(mock_figure.called)

    def test_correct_visualization_is_chosen_numerical(self):
        with mock.patch.object(NumericalStatisticalCheck, 'column_figure') as mock_figure:
            TextMetadataStatisticalCheck.metadata_figure('text', NumCharsMetadata(), None, None)
        self.assertTrue(mock_figure.called)
