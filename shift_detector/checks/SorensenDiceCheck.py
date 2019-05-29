from collections import defaultdict
from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.SorensenDicePrecalculation import SorensenDicePrecalculations
from shift_detector.precalculations.NGram import NGramType


class SorensenDiceCheck(Check):

    def __init__(self, ngram_type=NGramType.character, n=3):
        self.ngram_type = ngram_type
        self.n = n

    def run(self, store):
        data = store[SorensenDicePrecalculations(ngram_type=self.ngram_type, n=self.n)]

        examined_columns = set(data.keys())
        shifted_columns = set()
        explanation = defaultdict(str)

        for column_name in data:
            explanation[column_name] = "\tBaseline in Dataset1: {}\n" \
                                       "\tBaseline in Dataset2: {}\n" \
                                       "\tSorensen Dice Coefficient between Datasets: {}"\
                .format(data[column_name][0], data[column_name][1], data[column_name][2])

            if (abs(data[column_name][0] - data[column_name][1]) > 0.1 or
                    abs(data[column_name][0] - data[column_name][2]) > 0.1 or
                    abs(data[column_name][1] - data[column_name][2]) > 0.1):
                shifted_columns.add(column_name)

        return Report(examined_columns, shifted_columns, dict(explanation))