import unittest

from shift_detector.precalculations.Tokenizer import TokenizeIntoWords

class TestTokenizerFunctions(unittest.TestCase):

    def test_tokenize_into_words(self):
        normal = "This. is a'n example, ,, 12  35,6  , st/r--ing    \n test."
        empty = ""
        punctuation = ".  , * (  \n \t [}"
        tokenize_into_words = TokenizeIntoWords().tokenize_into_words
        self.assertEqual(tokenize_into_words(normal), ['this', 'is', "a'n", 'example', '12', '356', 'str', 'ing', 'test'])
        self.assertEqual(tokenize_into_words(empty), [])
        self.assertEqual(tokenize_into_words(punctuation), [])