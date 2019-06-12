import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.detector import Detector
from shift_detector.checks.statistical_checks.text_metadata_statistical_check import TextMetadataStatisticalCheck
from shift_detector.precalculations.store import Store
from shift_detector.precalculations.text_metadata import NumCharsMetadata, NumWordsMetadata, DistinctWordsRatioMetadata, \
    LanguagePerParagraph, UnknownWordRatioMetadata, StopwordRatioMetadata


class TestTextMetadataStatisticalCheck(unittest.TestCase):

    def setUp(self):
        self.poems = [
            'Tell me not, in mournful numbers,\nLife is but an empty dream!\nFor the soul is dead that slumbers,\nAnd things are not what they seem.',
            'Life is real! Life is earnest!\nAnd the grave is not its goal;\nDust thou art, to dust returnest,\nWas not spoken of the soul.',
            'Not enjoyment, and not sorrow,\nIs our destined end or way;\nBut to act, that each to-morrow\nFind us farther than to-day.',
            'Art is long, and Time is fleeting,\nAnd our hearts, though stout and brave,\nStill, like muffled drums, are beating\nFuneral marches to the grave.',
            'In the world’s broad field of battle,\nIn the bivouac of Life,\nBe not like dumb, driven cattle!\nBe a hero in the strife! ',
            'Trust no Future, howe’er pleasant!\nLet the dead Past bury its dead!\nAct,— act in the living Present!\nHeart within, and God o’erhead! ',
            'LIFE, believe, is not a dream\nSo dark as sages say;\nOft a little morning rain\nForetells a pleasant day.\nSometimes there are clouds of gloom,\nBut these are transient all;\nIf the shower will make the roses bloom,\nO why lament its fall ? ',
            "Rapidly, merrily,\nLife's sunny hours flit by,\nGratefully, cheerily,\nEnjoy them as they fly !",
            "What though Death at times steps in\nAnd calls our Best away ?\nWhat though sorrow seems to win,\nO'er hope, a heavy sway ?\nYet hope again elastic springs,\nUnconquered, though she fell;\nStill buoyant are her golden wings,\nStill strong to bear us well.\nManfully, fearlessly,\nThe day of trial bear,\nFor gloriously, victoriously,\nCan courage quell despair ! ",
            'When sinks my heart in hopeless gloom,\nAnd life can shew no joy for me;\nAnd I behold a yawning tomb,\nWhere bowers and palaces should be;\nIn vain you talk of morbid dreams;\nIn vain you gaily smiling say,\nThat what to me so dreary seems,\nThe healthy mind deems bright and gay.']
        self.phrases = ['Front-line leading edge website',
                        'Upgradable upward-trending software',
                        'Virtual tangible throughput',
                        'Robust secondary open system',
                        'Devolved multimedia knowledge user',
                        'Intuitive encompassing alliance',
                        'Automated 3rd generation benchmark',
                        'Switchable global info-mediaries',
                        'Automated 24/7 alliance',
                        'Down-sized homogeneous software']

    def test_significant_metadata(self):
        pvalues = pd.DataFrame([[0.001, 0.2]], columns=['num_chars', 'distinct_words_ratio'], index=['pvalue'])
        result = TextMetadataStatisticalCheck(significance=0.01).significant_metadata(pvalues)
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
