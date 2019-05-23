import pandas as pd
from shift_detector.preprocessors.NGram import NGram, NGramType
from shift_detector.preprocessors.Store import Store
from shift_detector.checks.Check import Check, Report
from datawig.utils import random_split


class SorensenDiceReport(Report):
    def __init__(self, result: dict):
        self.result = result

    def print_report(self):
        print('Sorensen Dice Report')
        for i in self.result:
            print(i, self.result[i])


class SorensenDiceCheck(Check):

    @staticmethod
    def count_fragments(ngram: dict) -> int:
        """
        Counts the the fragments in an ngram
        :param ngram:
        :return: the number of fragments as int
        """
        n = 0
        for i in ngram:
            n += ngram[i]

        return n

    @staticmethod
    def calculate_sdc(histo1: dict, histo2: dict) -> float:
        """
        Calculates the Sørensen Dice Coefficient between the two histograms
        :param histo1:
        :param histo2:
        :return: the Sørensen Dice Coefficient as float
        """
        nt = 0
        for i in histo1:
            if i in histo2:
                nt += min(histo1[i], histo2[i])

        return 2 * nt / (SorensenDiceCheck.count_fragments(histo1) + SorensenDiceCheck.count_fragments(histo2))

    @staticmethod
    def join_and_normalize_ngrams(ser: pd.Series) -> dict:
        """
        Join and normalize all ngrams to one ngram
        :param ser: A series that contains the ngrams to join
        :return: A dict that represents the final ngram
        """
        final_ngram = {}
        for ngram in ser:
            for j in ngram:
                final_ngram[j] = final_ngram[j] + ngram[j] if j in final_ngram else ngram[j]
        n = len(ser)
        for i in final_ngram:
            final_ngram[i] /= n
        return final_ngram

    def run(self, store: Store) -> SorensenDiceReport:
        """
        Calculate the Sørensen dice coefficient between two columns
        :param store:
        :return: CheckResult
        """

        df1, df2 = store[NGram(n=3, ngram_type=NGramType.character)]

        df1a, df1b = random_split(df1, [0.95, 0.05], seed=11)           # Baseline for df1
        df2a, df2b = random_split(df2, [0.95, 0.05], seed=11)           # Baseline for df2

        result = {}
        for i in df1:

            result[i] = (self.calculate_sdc(self.join_and_normalize_ngrams(df1a[i]),
                                            self.join_and_normalize_ngrams(df1b[i])),
                         self.calculate_sdc(self.join_and_normalize_ngrams(df2a[i]),
                                            self.join_and_normalize_ngrams(df2b[i])),
                         self.calculate_sdc(self.join_and_normalize_ngrams(df1[i]),
                                            self.join_and_normalize_ngrams(df2[i])))

        return SorensenDiceReport(result)
