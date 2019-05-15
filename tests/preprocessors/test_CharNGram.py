import unittest
import pandas as pd
from shift_detector.preprocessors.CharNGram import CharNGram


class TestCharNGram(unittest.TestCase):

    def setUp(self):
        self.ser1 = pd.Series(['Hello World', 'Foo Bar Baz'])
        self.ser2 = pd.Series(['TestText1', 'TestText2'])
        self.n = 5
        self.ng = CharNGram(self.n)
        self.res1, self.res2 = self.ng.process(self.ser1, self.ser2)

    def test_result_length(self):
        self.assertEqual(len(self.res1[0]), len(self.ser1[0]) - self.n + 1)
        self.assertEqual(len(self.res1[1]), len(self.ser1[1]) - self.n + 1)
        self.assertEqual(len(self.res2[0]), len(self.ser2[0]) - self.n + 1)
        self.assertEqual(len(self.res2[1]), len(self.ser2[1]) - self.n + 1)

    def test_result_content(self):
        self.assertIn('hello', self.res1[0])
        self.assertIn('oo ba', self.res1[1])
        self.assertIn('text2', self.res2[1])

    def test_eq(self):
        ng2 = CharNGram(6)
        ng3 = CharNGram(6)
        self.assertTrue(ng2 == ng3)
        self.assertFalse(self.ng == ng2)

    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: CharNGram(0))
        self.assertRaises(ValueError, lambda: CharNGram(-1))

    def test_hash(self):
        ng2 = CharNGram(self.n)
        self.assertEqual(hash(self.ng), hash(ng2))
