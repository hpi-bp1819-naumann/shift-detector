import unittest
from unittest.mock import patch
from shift_detector.preprocessors.NGram import NGram
import pandas as pd


class TestNGram(unittest.TestCase):

    @patch.multiple(NGram, __abstractmethods__=set())
    def setUp(self):
        self.ng1 = NGram(5)
        self.ng2 = NGram(6)
        self.ng3 = NGram(6)
        self.ser1 = pd.Series(['Hello World', 'Foo Bar Baz'])
        self.ser2 = pd.Series(['TestText1', 'TestText2'])

    def test_eq(self):
        self.assertTrue(self.ng2 == self.ng3)
        self.assertFalse(self.ng1 == self.ng2)

    @patch.multiple(NGram, __abstractmethods__=set())
    def test_exception_on_small_n(self):
        self.assertRaises(ValueError, lambda: NGram(0))
        self.assertRaises(ValueError, lambda: NGram(-1))

    def test_hash(self):
        self.assertEqual(hash(self.ng2), hash(self.ng3))

    def test_process(self):
        res1, res2 = self.ng1.process(self.ser1, self.ser2)
        self.assertListEqual(list(res1), list(self.ser1))
        self.assertListEqual(list(res2), list(self.ser2))