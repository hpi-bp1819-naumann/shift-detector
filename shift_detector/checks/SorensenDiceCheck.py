import pandas as pd
from shift_detector.preprocessors.NGram import NGram, NGramType
from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check, Report


class SorensenDiceReport(Report):
    def __init__(self, distances):
        self.distances = distances

    def print_report(self):
        print('### SORENCEN DICE REPORT ###')
        for i in self.distances:
            print(i, self.distances[i])


class SorensenDiceCheck(Check):

    @staticmethod
    def join_ngrams(ngram1, ngram2):
        for i in ngram2:
            ngram1[i] = ngram1[i] + ngram2[i] if i in ngram1 else ngram2[i]

    @staticmethod
    def count_fragments(ngram):
        n = 0
        for i in ngram:
            n += ngram[i]

        return n

    @staticmethod
    def calculate_sdc(histo1, histo2):
        nt = 0
        for i in histo1:
            if i in histo2:
                nt += min(histo1[i], histo2[i])

        return 2 * nt / (SorensenDiceCheck.count_fragments(histo1) + SorensenDiceCheck.count_fragments(histo2))

    def run(self, store) -> SorensenDiceReport:
        """
        Calculate the sorensen dice coefficient between two columns
        :param columns:
        :return: CheckResult
        """

        df1, df2 = store[NGram(n=5, ngram_type=NGramType.character)]

        result = {}
        for i in store.columns:
            ngram1 = {}
            ngram2 = {}
            for j in df1[i]:
                self.join_ngrams(ngram1, j)
            for j in df2[i]:
                self.join_ngrams(ngram2, j)
            result[i] = self.calculate_sdc(ngram1, ngram2)

        return SorensenDiceReport(result)
