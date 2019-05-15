import unittest
import pandas as pd
from shift_detector.preprocessors.WordNGram import WordNGram


class TestWordNGram(unittest.TestCase):

    def setUp(self):
        self.ser1 = pd.Series(['This is a text', 'This is another text'])
        self.ser2 = pd.Series(['test text test text', 'test test test test test'])
        self.n1 = 1
        self.wh1 = WordNGram(self.n1)
        self.n2 = 3
        self.wh2 = WordNGram(self.n2)
        self.res11, self.res12 = self.wh1.process(self.ser1, self.ser2)
        self.res21, self.res22 = self.wh2.process(self.ser1, self.ser2)

    def test_result_length(self):
        self.assertEqual(len(self.res11[0]), 4)
        self.assertEqual(len(self.res21[1]), 2)
        self.assertEqual(len(self.res12[0]), 2)
        self.assertEqual(len(self.res22[1]), 1)

    def test_result_content(self):
        self.assertIn(('this',), self.res11[0])
        self.assertIn(('another',), self.res11[1])
        self.assertIn(('is', 'another', 'text'), self.res21[1])
        self.assertEqual(self.res22[1][('test', 'test', 'test')], 3)

    def test_eq(self):
        wh3 = WordNGram(3)
        self.assertTrue(self.wh2 == wh3)
        self.assertFalse(self.wh1 == self.wh2)

    def test_exception_on_small_n(self):
        self.assertRaises(Exception, lambda: WordNGram(0))
        self.assertRaises(Exception, lambda: WordNGram(-1))

    def test_hash(self):
        wh3 = WordNGram(self.n2)
        self.assertIsNotNone(hash(self.wh2), hash(wh3))
