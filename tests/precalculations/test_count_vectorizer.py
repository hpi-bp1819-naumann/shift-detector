import unittest
from shift_detector.precalculations.count_vectorizer import CountVectorizer
from shift_detector.precalculations.store import Store
import pandas as pd
import numpy as np


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
        test_dict1 = {'col1': np.array([[1,0] if not i == 9 else [0,1] for i in range(10)])}
        test_dict2 = {'col1': np.array([[0,1] if not i == 9 else [1,0] for i in range(10)])}

        self.assertEqual(res1.keys(), test_dict1.keys())
        for val1, test_val1 in zip(res1.values(), test_dict1.values()):
            for row1, test_row1 in zip(val1, test_val1):
                np.testing.assert_equal(row1, test_row1)
        self.assertEqual(res2.keys(), test_dict2.keys())
        for val2, test_val2 in zip(res2.values(), test_dict2.values()):
            for row2, test_row2 in zip(val2, test_val2):
                np.testing.assert_equal(row2, test_row2)
