import numpy as np
from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.utils.column_management import ColumnType
from collections import Counter


class LdaCheck(Check):

    def __init__(self, significance=10, n_topics=20, n_iter=10, lib='sklearn', random_state=0,
                 cols=None, trained_model=None, stop_words='english', max_features=None):
        self.significance = None
        if isinstance(significance, int) and significance > 0:
            self.significance = significance
        else:
            raise ValueError('Please enter a positive int as significance')
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.lib = lib
        self.random_state = random_state
        self.cols = cols
        self.trained_model = trained_model
        self.stop_words = stop_words
        self.max_features = max_features

    def run(self, store) -> Report:
        df1_texts, df2_texts = store[ColumnType.text]

        shifted_columns = set()
        explanation = {}

        if self.cols is None:
            col_names = df1_texts.columns
            self.cols = list(col_names)
        else:
            if isinstance(self.cols, str):
                if self.cols in df1_texts.columns:
                    col_names = self.cols
            else:
                for col in self.cols:
                    if col not in df1_texts.columns:
                        raise ValueError('Given column is not contained in given datasets')
                col_names = self.cols

        df1_embedded, df2_embedded = store[LdaEmbedding(n_topics=self.n_topics, n_iter=self.n_iter, lib=self.lib,
                                                        random_state=self.random_state, cols=self.cols,
                                                        trained_model=self.trained_model, stop_words=self.stop_words,
                                                        max_features=self.max_features)]

        for col in col_names:
            count_topics1 = Counter(df1_embedded['topics ' + col])
            count_topics2 = Counter(df2_embedded['topics ' + col])

            labels1_ordered, values1_ordered = zip(*sorted(count_topics1.items(), key=lambda kv: kv[0]))
            values1_perc = [x * 100 / np.array(values1_ordered).sum() for x in values1_ordered]

            labels2_ordered, values2_ordered = zip(*sorted(count_topics2.items(), key=lambda kv: kv[0]))
            values2_perc = [x * 100 / np.array(values2_ordered).sum() for x in values2_ordered]

            for i, (v1, v2) in enumerate(zip(values1_perc, values2_perc)):
                # number of rounded digits is 1 per default
                if abs(round(v1 - v2, 1)) >= self.significance:
                    shifted_columns.add(col)
                    explanation['Topic '+str(i)+' diff in column '+col] = round(v1 - v2, 1)

        return Report(check_name='LDA Check', examined_columns=col_names, shifted_columns=shifted_columns, explanation=explanation)
