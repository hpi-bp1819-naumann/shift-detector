import unittest
from unittest import mock

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.checks.statistical_checks.categorical_statistical_check import CategoricalStatisticalCheck
from shift_detector.checks.statistical_checks.numerical_statistical_check import NumericalStatisticalCheck
from shift_detector.checks.statistical_checks.text_metadata_statistical_check import TextMetadataStatisticalCheck
from shift_detector.detector import Detector
from shift_detector.precalculations.store import Store
from shift_detector.precalculations.text_metadata import NumCharsMetadata, NumWordsMetadata, \
    DistinctWordsRatioMetadata, LanguagePerParagraph, UnknownWordRatioMetadata, StopwordRatioMetadata, LanguageMetadata

import tests.test_data as td


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

    def test_correct_visualization_is_chosen_categorical(self):
        with mock.patch.object(CategoricalStatisticalCheck, 'column_figure') as mock_figure:
            TextMetadataStatisticalCheck.metadata_figure('text', LanguageMetadata(), None, None)
        self.assertTrue(mock_figure.called)  # replacement for 3.5
        # mock_figure.assert_called_once_with(('text', LanguageMetadata().metadata_name()), None, None)

    def test_correct_visualization_is_chosen_numerical(self):
        with mock.patch.object(NumericalStatisticalCheck, 'column_figure') as mock_figure:
            TextMetadataStatisticalCheck.metadata_figure('text', NumCharsMetadata(), None, None)
        self.assertTrue(mock_figure.called)  # replacement for 3.5
        # mock_figure.assert_called_once_with(('text', NumCharsMetadata().metadata_name()), None, None)
