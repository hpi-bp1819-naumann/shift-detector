import numpy as np
from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.utils.column_management import ColumnType
from collections import Counter


class LdaCheck(Check):

    def __init__(self, significance=10, cols=None):
        self.significance = None
        self.cols = None
        if cols and (not isinstance(cols, list) or any(not isinstance(col, str) for col in cols)):
            raise TypeError('Cols has to be list of strings')
        else:
            self.cols = cols
        if isinstance(significance, int) and significance > 0:
            self.significance = significance
        else:
            raise ValueError('Please enter a positive int as significance')

    def run(self, store) -> Report:
        # TODO make CountVectorizer accessible from this interface
        df1_texts, df2_texts = store[ColumnType.text]
        # TODO make LDAEmbedding accessible here with additional parameters, currently a trained model will always be \
        #  used and the embeddings may be completely unrelated 
        df1_embedded, df2_embedded = store[LdaEmbedding(cols=self.cols)]
        shifted_columns = set()
        explanation = {}
        col_names = df1_texts.columns

        for col in col_names:
            count_topics1 = Counter(df1_embedded['topics ' + col])
            count_topics2 = Counter(df2_embedded['topics ' + col])

            labels1_ordered, values1_ordered = zip(*sorted(count_topics1.items(), key=lambda kv: kv[0]))
            values1_perc = [x * 100 / np.array(values1_ordered).sum() for x in values1_ordered]

            labels2_ordered, values2_ordered = zip(*sorted(count_topics2.items(), key=lambda kv: kv[0]))
            values2_perc = [x * 100 / np.array(values2_ordered).sum() for x in values2_ordered]

            for i, (v1, v2) in enumerate(zip(values1_perc, values2_perc)):
                if abs(round(v1 - v2, 2)) >= self.significance:
                    shifted_columns.add(col)
                    explanation['Topic '+str(i)+' diff in column '+col] = round(v1 - v2, 2)

        return Report(col_names, shifted_columns, explanation)
