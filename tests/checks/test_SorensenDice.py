import unittest
from unittest.mock import MagicMock
from shift_detector.checks.SorensenDiceCheck import SorensenDiceCheck, SorensenDiceReport
from shift_detector.preprocessors.Store import Store
import pandas as pd
import io


class TestSorensenDice(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'col1': ['ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef',
                                          'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef', 'ab cd ef']})
        self.df2 = pd.DataFrame({'col1': ['ab ','hi ','jk ','lm ','no ','pq ','rs ','tu ','vw ','xy ','z1 ','23 ','45 ','67 ','89 ']})

        self.store = Store(self.df1, self.df2)
        self.result = SorensenDiceCheck().run(self.store)

    def test_result(self):
        self.assertEqual(len(self.result.result), 1)
        self.assertEqual(self.result.result['col1'], (1.0, 0.0, (2 / 15) / 7))   # ngrams get normalized during join

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_print_result(self, mock_stdout):
        self.result.print_report()
        output = mock_stdout.getvalue()
        self.assertIn('Sorensen Dice Report', output)


    def test_count_fragments(self):
        ngram1 = {'abc': 6, 'bcd':4, 'def': 3}
        self.assertEqual(SorensenDiceCheck.count_fragments(ngram1), 13)

    def test_calculate_sdc(self):
        ngram1 = {'abc': 6, 'bcd':4, 'def': 3}
        ngram2 = {'abc': 2, 'def': 3, 'efg': 4}
        self.assertEqual(SorensenDiceCheck.calculate_sdc(ngram1, ngram2), 10 / 22)

    def test_join_and_normalze_ngrams(self):
        ngram_ser = pd.Series([{'abc': 6, 'bcd':4, 'def': 3}, {'abc': 2, 'def': 3, 'efg': 4}])
        result = SorensenDiceCheck.join_and_normalize_ngrams(ngram_ser)
        self.assertDictEqual(result, {'abc': 4, 'bcd': 2, 'def': 3, 'efg': 2})
        self.assertDictEqual(SorensenDiceCheck.join_and_normalize_ngrams(pd.Series([])), {})
