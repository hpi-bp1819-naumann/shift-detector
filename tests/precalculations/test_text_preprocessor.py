import unittest

from pandas.util.testing import assert_frame_equal

from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.TextPreprocessor import *


class TestTextPreprocessorFunctions(unittest.TestCase):

    def setUp(self):
        poems = [
            'Tell me not, in mournful numbers,\nLife is but an empty dream!\nFor the soul is dead that slumbers,\n'
            'And things are not what they seem.',
            'Life is real! Life is earnest!\nAnd the grave is not its goal;\nDust thou art, to dust returnest,\n'
            'Was not spoken of the soul.'
        ]
        phrases = [
            'Front-line leading edge website',
            'Upgradable upward-trending software'
        ]
        df1 = pd.DataFrame.from_dict({'text': poems})
        df2 = pd.DataFrame.from_dict({'text': phrases})
        self.store = Store(df1, df2)

    def test_tokenize_into_words_preprocessor(self):
        md1, md2 = TokenizeIntoLowerWordsPrecalculation().process(self.store)
        sol1_1 = ['tell', 'me', 'not', 'in', 'mournful', 'numbers', 'life', 'is', 'but', 'an', 'empty', 'dream', 'for',
                  'the', 'soul', 'is', 'dead', 'that', 'slumbers', 'and', 'things', 'are', 'not', 'what', 'they', 'seem'
                  ]
        sol1_2 = ['life', 'is', 'real', 'life', 'is', 'earnest', 'and', 'the', 'grave', 'is', 'not', 'its', 'goal',
                  'dust', 'thou', 'art', 'to', 'dust', 'returnest', 'was', 'not', 'spoken', 'of', 'the', 'soul']
        sol2_1 = ['front', 'line', 'leading', 'edge', 'website']
        sol2_2 = ['upgradable', 'upward', 'trending', 'software']
        solution1 = pd.DataFrame([[sol1_1], [sol1_2]], columns=['text'])
        solution2 = pd.DataFrame([[sol2_1], [sol2_2]], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_tokenize_into_words_function(self):
        normal = "This. is a'n example, ,, 12  35,6  , st/r--ing    \n test."
        empty = ""
        punctuation = ".  , * (  \n \t [}"
        tokenize_into_words = TokenizeIntoLowerWordsPrecalculation().tokenize_into_words
        self.assertEqual(tokenize_into_words(normal),
                         ['this', 'is', "a'n", 'example', '12', '356', 'str', 'ing', 'test'])
        self.assertEqual(tokenize_into_words(empty), [])
        self.assertEqual(tokenize_into_words(punctuation), [])
