import unittest
from shift_detector.precalculations.count_vectorizer import CountVectorizer
from shift_detector.precalculations.store import Store
import pandas as pd
import numpy as np


class TestCountVectorizer(unittest.TestCase):

    def setUp(self):
        self.count1 = CountVectorizer(columns=['col1'], stop_words='english', max_features=2)
        self.count2 = CountVectorizer(columns=['col1'], stop_words='english', max_features=2)
        self.count3 = CountVectorizer(columns=['col1'], stop_words='english', max_features=3)

        self.df1 = pd.DataFrame({'col1':
                                ['duck', 'duck', 'duck', 'duck', 'duck',
                                 'duck', 'duck', 'duck', 'duck', 'goose']})
        self.df2 = pd.DataFrame({'col1':
                                ['goose', 'goose', 'goose', 'goose', 'goose',
                                 'goose', 'goose', 'goose', 'goose', 'duck']})
        self.store = Store(self.df1, self.df2)

    def test_eq(self):
        self.assertEqual(self.count1, self.count2)
        self.assertNotEqual(self.count1, self.count3)

    def test_exception_for_max_features(self):
        self.assertRaises(ValueError, lambda: CountVectorizer(columns=[''], max_features=0))
        self.assertRaises(TypeError, lambda: CountVectorizer(columns=[''], max_features=3.5))

    def test_exception_for_stop_words(self):
        self.assertRaises(Exception, lambda: CountVectorizer(columns=[''], stop_words='abcd'))
        self.assertRaises(Exception, lambda: CountVectorizer(columns=[''], stop_words=['english', ' abcd']))
        self.assertRaises(TypeError, lambda: CountVectorizer(columns=[''], stop_words=['english', 42]))

    def test_hash(self):
        self.assertEqual(hash(self.count1), hash(self.count2))
        self.assertNotEqual(hash(self.count1), hash(self.count3))

    def test_process(self):
        res1, res2, feature_names, all_vecs = self.count1.process(self.store)
        expected_dict1 = {'col1': np.array([[1,0] if not i == 9 else [0,1] for i in range(10)])}
        expected_dict2 = {'col1': np.array([[0,1] if not i == 9 else [1,0] for i in range(10)])}

        self.assertEqual(res1.keys(), expected_dict1.keys())
        np.testing.assert_equal(res1['col1'], expected_dict1['col1'])

        self.assertEqual(res2.keys(), expected_dict2.keys())
        np.testing.assert_equal(res2['col1'], expected_dict2['col1'])

