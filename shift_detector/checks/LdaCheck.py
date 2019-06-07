import pandas as pd
import numpy as np
from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.LdaEmbedding import LdaEmbedding
from shift_detector.Utils import ColumnType
from collections import Counter


class LdaCheck(Check):

    def __init__(self, significance=10):
        self.significance = None
        if isinstance(significance, int) and significance > 0:
            self.significance = significance
        else:
            raise ValueError('Please enter a positive int as significance')

    def run(self, store) -> Report:
        # TODO make CountVectorizer accessible from this interface
        processed_df1, processed_df2 = store[LdaEmbedding()]

        count_topics1 = Counter(processed_df1['topic'])
        count_topics2 = Counter(processed_df2['topic'])

        labels1_ordered, values1_ordered = zip(*sorted(count_topics1.items(), key=lambda kv: kv[0]))
        values1_perc = [x * 100 / np.array(values1_ordered).sum() for x in values1_ordered]

        labels2_ordered, values2_ordered = zip(*sorted(count_topics2.items(), key=lambda kv: kv[0]))
        values2_perc = [x * 100 / np.array(values2_ordered).sum() for x in values2_ordered]

        shifted_columns = set()
        explanation = dict()

        for i, (v1, v2) in enumerate(zip(values1_perc, values2_perc)):
            if abs(round(v1 - v2, 1)) >= self.significance:
                explanation['Topic ' + str(i) + ' diff'] = round(v1 - v2, 1)

        if explanation != dict():
            # currently shows all text columns as shifted because they are merged together
            shifted_columns = store[ColumnType.text]

        return Report(store[ColumnType.text], shifted_columns, explanation)