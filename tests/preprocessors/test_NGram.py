import unittest
import pandas as pd
from shift_detector.preprocessors.NGram import NGram


class TestNGram(unittest.TestCase):

    def setUp(self):
        self.n = 5
        self.ng = NGram(self.n)

    def test_eq(self):
        ng2 = NGram(6)
        ng3 = NGram(6)
        self.assertTrue(ng2 == ng3)
        self.assertFalse(self.ng == ng2)

    def test_exception_on_small_n(self):
        self.assertRaises(Exception, lambda: NGram(0))
        self.assertRaises(Exception, lambda: NGram(-1))

    def test_hash(self):
        ng2 = NGram(self.n)
        self.assertEqual(hash(self.ng), hash(ng2))
