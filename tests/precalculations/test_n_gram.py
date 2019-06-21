import unittest
from Morpheus.precalculations.n_gram import NGram, NGramType
from Morpheus.precalculations.store import Store
import pandas as pd


class TestNGram(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab ', 'hi ', 'jk ', 'lm ', 'no ', 'pq ', 'rs ', 'tu ', 'vw ', 'xy ', 'z1 ',
                                          '23 ', '45 ', '67 ', '89 ']})

        self.wordng1 = NGram(1, NGramType.word)
        self.wordng2 = NGram(1, NGramType.word)
        self.wordng3 = NGram(2, NGramType.word)
        self.charng1 = NGram(1, NGramType.character)
        self.charng2 = NGram(5, NGramType.character)

        self.store = Store(self.df1, self.df2)

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
        res1, res2 = self.wordng1.process(self.store)
        self.assertListEqual(list(res1['col1']), [{('ab',): 1, ('cd',): 1, ('ef',): 1}] * 20)
        res3, res4 = self.charng2.process(self.store)
        self.assertListEqual(list(res3['col1']), [{('a', 'b', ' ', 'c', 'd'): 1, ('b', ' ', 'c', 'd', ' '): 1,
                                                   (' ', 'c', 'd', ' ', 'e'): 1, ('c', 'd', ' ', 'e', 'f'): 1}] * 20)
