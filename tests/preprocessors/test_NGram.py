import unittest
from shift_detector.preprocessors.NGram import NGram, NGramType
import pandas as pd


class TestNGram(unittest.TestCase):

    def setUp(self):
        self.wordng1 = NGram(1, NGramType.word)
        self.wordng2 = NGram(1, NGramType.word)
        self.wordng3 = NGram(2, NGramType.word)
        self.charng1 = NGram(1, NGramType.character)
        self.charng2 = NGram(5, NGramType.character)
        self.ser1 = pd.Series(['Hello World', 'Foo Bar Baz'])
        self.ser2 = pd.Series(['TestText1', 'TestText2'])
        self.ser3 = pd.Series(['abababa'])

    def test_eq(self):
        self.assertTrue(self.wordng1 == self.wordng2)
        self.assertFalse(self.wordng1 == self.wordng3)
        self.assertFalse(self.wordng1 == self.charng1)

    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: NGram(0, NGramType.word))
        self.assertRaises(ValueError, lambda: NGram(-1, NGramType.character))

    def test_hash(self):
        self.assertEqual(hash(self.wordng1), hash(self.wordng2))

    def test_generate_ngram(self):
        self.assertDictEqual(self.charng1.generate_ngram(('t', 'e', 's', 't')), {('t',): 2, ('e',): 1, ('s',): 1})

    def test_process(self):
        res1, res2 = self.wordng1.process(self.ser1, self.ser2)
        self.assertListEqual(list(res2), [{('testtext1',): 1}, {('testtext2',): 1}])
        res3, res4 = self.charng2.process(self.ser1, self.ser3)
        self.assertListEqual(list(res4), [{('a', 'b', 'a', 'b', 'a'): 2,
                                           ('b', 'a', 'b', 'a', 'b'): 1}])
