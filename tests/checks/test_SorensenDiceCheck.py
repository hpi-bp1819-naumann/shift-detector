import unittest
from shift_detector.checks.SorensenDiceCheck import SorensenDiceCheck
from shift_detector.precalculations.NGram import NGramType
from shift_detector.precalculations.Store import Store
import pandas as pd


class TestSorensenDiceCheck(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab ','hi ','jk ','lm ','no ','pq ','rs ','tu ','vw ','xy ','z1 ','23 ','45 ','67 ','89 ']})

        self.store = Store(self.df1, self.df2)
        self.result = SorensenDiceCheck(ngram_type=NGramType.character, n=3).run(self.store)

    def test_examined_columns(self):
        self.assertEqual(self.result.examined_columns, {'col1'})

    def test_shifted_columns(self):
        self.assertEqual(self.result.shifted_columns, {'col1'})

    def test_explanation_existence(self):
        self.assertNotEqual(self.result.explanation, '')
