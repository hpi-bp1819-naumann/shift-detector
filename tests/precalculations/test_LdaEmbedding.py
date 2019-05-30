import unittest
from shift_detector.precalculations.LdaEmbedding import LdaEmbedding
import pandas as pd


class TestLdaEmbedding(unittest.TestCase):

    def setUp(self):
        self.lda1 = LdaEmbedding(n_topics=2, n_iter=1)
        self.lda2 = LdaEmbedding(n_topics=2, n_iter=1)

        self.ser1 = pd.Series(['Hello World', 'Foo Bar Baz'])
        self.ser2 = pd.Series(['TestText1', 'TestText2'])
        self.ser3 = pd.Series(['abababa'])

    def test_eq(self):
        self.assertTrue(self.lda1 == self.lda2)

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
