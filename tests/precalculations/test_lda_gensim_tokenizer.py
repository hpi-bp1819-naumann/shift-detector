import unittest
from shift_detector.precalculations.lda_gensim_tokenizer import LdaGensimTokenizer
from shift_detector.precalculations.store import Store
import pandas as pd


class TestLdaGensimTokenizer(unittest.TestCase):

    def setUp(self):
        self.token1 = LdaGensimTokenizer(cols='col1', stop_words='english')
        self.token2 = LdaGensimTokenizer(cols='col1', stop_words='english')
        self.token3 = LdaGensimTokenizer(cols='col1', stop_words='french')

        self.df1 = pd.DataFrame({'col1':
                                ['duck, duck, duck, duck, duck, \
                                  duck, duck, duck, duck, goose']*9 +
                                ['goose, goose, goose, goose, goose,\
                                  goose, goose, goose, goose, duck']
                                 })
        self.df2 = pd.DataFrame({'col1':
                                 ['goose, goose, goose, goose, goose,\
                                 goose, goose, goose, goose, duck']*9 +
                                 ['duck, duck, duck, duck, duck, \
                                  duck, duck, duck, duck, goose']
                                 })

        self.store = Store(self.df1, self.df2)

    def test_exception_for_cols(self):
        self.assertRaises(TypeError, lambda: LdaGensimTokenizer())
        self.assertRaises(TypeError, lambda: LdaGensimTokenizer(cols=None))
        self.assertRaises(TypeError, lambda: LdaGensimTokenizer(cols=42))

    def test_exception_for_stop_words(self):
        self.assertRaises(ValueError, lambda: LdaGensimTokenizer(cols='col1', stop_words=['english', 'abcd']))
        self.assertRaises(ValueError, lambda: LdaGensimTokenizer(cols='', stop_words='abcd'))
        self.assertRaises(ValueError, lambda: LdaGensimTokenizer(cols='', stop_words=['english', ' abcd']))
        self.assertRaises(TypeError, lambda: LdaGensimTokenizer(cols='', stop_words=['english', 42]))

    def test_eq(self):
        self.assertEqual(self.token1, self.token2)
        self.assertNotEqual(self.token1, self.token3)

    def test_hash(self):
        self.assertEqual(hash(self.token1), hash(self.token2))
        self.assertNotEqual(hash(self.token1), hash(self.token3))

    def test_process(self):
        res1, res2 = self.token1.process(self.store)

        pd.testing.assert_frame_equal(res1, pd.DataFrame.from_dict({'col1':
                                                                    [['duck', 'duck', 'duck', 'duck', 'duck',
                                                                      'duck', 'duck', 'duck', 'duck', 'goose']]*9 +
                                                                    [['goose', 'goose', 'goose', 'goose', 'goose',
                                                                      'goose', 'goose', 'goose', 'goose', 'duck']]
                                                                    }))

        pd.testing.assert_frame_equal(res2, pd.DataFrame.from_dict({'col1':
                                                                    [['goose', 'goose', 'goose', 'goose', 'goose',
                                                                      'goose', 'goose', 'goose', 'goose', 'duck']]*9 +
                                                                    [['duck', 'duck', 'duck', 'duck', 'duck',
                                                                      'duck', 'duck', 'duck', 'duck', 'goose']]
                                                                    }))


