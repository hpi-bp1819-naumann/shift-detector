import unittest
import pandas as pd
from shift_detector.preprocessors.DefaultEmbedding import DefaultEmbedding


class TestDefault(unittest.TestCase):

    def setUp(self):
        self.n = 5
        self.ng1 = DefaultEmbedding()
        self.ng2 = DefaultEmbedding()
        self.ser1 = pd.Series(['Hello World', 'Foo Bar Baz'])
        self.ser2 = pd.Series(['TestText1', 'TestText2'])

    def test_eq(self):
        self.assertTrue(self.ng1 == self.ng2)

    def test_hash(self):
        self.assertEqual(hash(self.ng1), hash(self.ng2))

    def test_process(self):
        res1, res2 = self.ng1.process(self.ser1, self.ser2)
        self.assertListEqual(list(res1), list(self.ser1))
        self.assertListEqual(list(res2), list(self.ser2))
