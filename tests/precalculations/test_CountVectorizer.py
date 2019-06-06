import unittest
from shift_detector.precalculations.CountVectorizer import CountVectorizer
from shift_detector.precalculations.Store import Store
from pandas.util.testing import assert_frame_equal
import pandas as pd


class TestCountVectorizer(unittest.TestCase):

    def setUp(self):
        self.count1 = CountVectorizer(stop_words='english', max_features=2)
        self.count2 = CountVectorizer(stop_words='english', max_features=2)
        self.count3 = CountVectorizer(stop_words='english', max_features=3)

        self.df1 = pd.DataFrame({'col1':
                                ['duck', 'duck', 'duck', 'duck', 'duck',
                                 'duck', 'duck', 'duck', 'duck', 'goose']})
        self.df2 = pd.DataFrame({'col1':
                                ['goose', 'goose', 'goose', 'goose', 'goose',
                                 'goose', 'goose', 'goose', 'goose', 'duck']})
        self.store = Store(self.df1, self.df2)

    def test_eq(self):
        self.assertTrue(self.count1 == self.count2)
        self.assertFalse(self.count1 == self.count3)

    def test_exception_for_max_features(self):
        self.assertRaises(ValueError, lambda: CountVectorizer(max_features=0))
        self.assertRaises(TypeError, lambda: CountVectorizer(max_features=3.5))

    def test_exception_for_stop_words(self):
        self.assertRaises(Exception, lambda: CountVectorizer(stop_words='abcd'))
        self.assertRaises(Exception, lambda: CountVectorizer(stop_words=['english', ' abcd']))
        self.assertRaises(TypeError, lambda: CountVectorizer(stop_words=['english', 42]))

    def test_hash(self):
        self.assertEqual(hash(self.count1), hash(self.count2))

    def test_process(self):
        res1, res2 = self.count1.process(self.store)

        assert_frame_equal(res1, pd.DataFrame([1]*9 + [0], columns=['col1']))
        assert_frame_equal(res2, pd.DataFrame([0]*9 + [1], columns=['col1']))
