import pandas as pd
from shift_detector.precalculations.NGram import NGram, NGramType
from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.Precalculation import Precalculation
from datawig.utils import random_split


class SorensenDicePrecalculations(Precalculation):

    def __init__(self, ngram_type=NGramType.character, n=3):
        self.ngram_type = ngram_type
        self.n = n

    def __eq__(self, other):
        return self.ngram_type == other.ngram_type and self.n == other.n

    def __hash__(self):
        return hash((self.ngram_type, self.n))

    @staticmethod
    def count_fragments(ngram: dict) -> int:
        """
        Counts, how much elements the ngram contains
        e.g. {'abc': 3, 'bcd': 2} contains 5 elements
        :param ngram:
        :return: the number of fragments as int
        """
        return sum(ngram.values())

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

        return 2 * nt / (SorensenDicePrecalculations.count_fragments(histo1) +
                         SorensenDicePrecalculations.count_fragments(histo2))

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

    def process(self, store: Store) -> dict:
        """
        Calculate the Sørensen dice coefficient between two columns
        :param store:
        :return: CheckResult
        """

        df1, df2 = store[NGram(n=self.n, ngram_type=self.ngram_type)]

        df1a, df1b = random_split(df1, [0.95, 0.05], seed=11)           # Baseline for df1
        df2a, df2b = random_split(df2, [0.95, 0.05], seed=11)           # Baseline for df2

        if df1b.empty or df2b.empty:
            raise ValueError('Dataset to small for split ratio or n={} to big'.format(self.n))

        result = {}
        for i in df1:

            result[i] = (self.calculate_sdc(self.join_and_normalize_ngrams(df1a[i]),
                                            self.join_and_normalize_ngrams(df1b[i])),
                         self.calculate_sdc(self.join_and_normalize_ngrams(df2a[i]),
                                            self.join_and_normalize_ngrams(df2b[i])),
                         self.calculate_sdc(self.join_and_normalize_ngrams(df1[i]),
                                            self.join_and_normalize_ngrams(df2[i])))

        return result
