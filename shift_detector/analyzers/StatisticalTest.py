import pandas as pd
import numpy as np

from shift_detector.analyzers.analyzer import Analyzer, AnalyzerResult
from shift_detector.utils.TextEmbedding import get_column_embedding, DIMENSIONS
import shift_detector.utils.TextMetadata as text_md
from scipy import stats
from datawig.utils import random_split


class StatResult(AnalyzerResult):

    def __init__(self, num_stats, cat_stats, text_embedding_stats, text_meta_stats, significance=0.01,
                 significant_dim_ratio=0.0):
        super().__init__()
        self.num = num_stats
        self.cat = cat_stats
        self.has_text_stats = True if len(text_embedding_stats) > 0 and len(text_meta_stats) > 0 else False
        self.embedding = text_embedding_stats
        self.text_meta = text_meta_stats
        self.significance = significance
        self.significant_dim_ratio = significant_dim_ratio

    def is_significant(self, base1_p, base2_p, p):
        return (base1_p > self.significance
                and base2_p > self.significance
                and p <= self.significance)

    def significant_num_columns(self):
        return set(column for column in self.num.columns if self.is_significant(self.num.loc['base1', column],
                                                                                self.num.loc['base2', column],
                                                                                self.num.loc['test', column]))

    def significant_cat_columns(self):
        return set(column for column in self.cat.columns if self.is_significant(self.cat.loc['base1', column],
                                                                                self.cat.loc['base2', column],
                                                                                self.cat.loc['test', column]))

    def significant_embedding_columns(self):
        if not self.has_text_stats:
            return set()
        ratios = {}
        try:
            for column in self.embedding.columns.levels[0]:
                column_dims = pd.Series(self.embedding[column], index=['base1', 'base2', 'test'])
                differing_dimensions = [dimension for dimension in column_dims.columns
                                        if self.is_significant(column_dims[dimension].loc['base1'],
                                                               column_dims[dimension].loc['base2'],
                                                               column_dims[dimension].loc['test'])]
                ratios[column] = len(differing_dimensions) / len(column_dims.columns)
            return set(column for column in self.embedding.columns.levels[0]
                       if ratios[column] > self.significant_dim_ratio)
        except Exception as e:
            print('Error: Failed to derive significant columns from text embedding stats', e)
            return set()

    def significant_text_metadata_columns(self):
        if not self.has_text_stats:
            return set()
        columns = set()
        try:
            for column in self.text_meta.columns.levels[0]:
                columns_metadata = pd.Series(self.text_meta[column], index=['base1', 'base2', 'test'])
                if any(self.is_significant(columns_metadata[md].loc['base1'],
                                           columns_metadata[md].loc['base2'],
                                           columns_metadata[md].loc['test']) for md in columns_metadata.columns):
                    columns.add(column)
            return columns
        except Exception as e:
            print('Error: Failed to derive significant columns from text metadata stats', e)
            return set()

    def significant_columns(self):
        return set().union(*[self.significant_num_columns(), self.significant_cat_columns(),
                             self.significant_embedding_columns(), self.significant_text_metadata_columns()])


def has_same_value_set(column, part1, part2):
    return not set(part1[column].unique()).symmetric_difference(set(part2[column].unique()))


def is_unique(column, part1, part2):
    return len(part1[column].unique()) == len(part1[column]) and len(part2[column].unique()) == len(part2[column])


def has_reasonable_length(column, part1, part2, min_length=2):
    return text_md.min_num_words(part1[column].dropna()) > min_length \
           and text_md.min_num_words(part2[column].dropna()) > min_length


def chi_test(column, part1, part2):
    observed = pd.DataFrame.from_dict({'a': part1[column].value_counts(), 'b': part2[column].value_counts()})
    observed['a'] = observed['a'].add(5, fill_value=0)  # rule of succession
    observed['b'] = observed['b'].add(5, fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(observed, lambda_='log-likelihood')
    return p


def ks_test(column, part1, part2):
    ks_test_result = stats.ks_2samp(part1[column], part2[column])
    return ks_test_result.pvalue


def ks_test_series(series1, series2):
    ks_test_result = stats.ks_2samp(series1, series2)
    return ks_test_result.pvalue


class StatisticalTestAnalyzer(Analyzer):

    def __init__(self, data1, data2):
        Analyzer.__init__(self, data1, data2)
        self.data1_p1, self.data1_p2 = random_split(self.data1)
        self.data2_p1, self.data2_p2 = random_split(self.data2)

    def analyze_categorical_columns(self, categorical):
        cat_stats = pd.DataFrame()
        for column in categorical:
            base1_p = chi_test(column, self.data1_p1, self.data1_p2)
            base2_p = chi_test(column, self.data2_p1, self.data2_p2)
            p = chi_test(column, self.data1, self.data2)
            cat_stats[column] = [base1_p, base2_p, p]
            cat_stats.index = ['base1', 'base2', 'test']
        return cat_stats

    def analyze_numerical_columns(self, numerical):
        num_stats = pd.DataFrame()
        for column in numerical:
            base1_p = ks_test(column, self.data1_p1, self.data1_p2)
            base2_p = ks_test(column, self.data2_p1, self.data2_p2)
            p = ks_test(column, self.data1, self.data2)
            num_stats[column] = [base1_p, base2_p, p]
            num_stats.index = ['base1', 'base2', 'test']
        return num_stats

    def analyze_text_embedding(self, text):
        index = pd.MultiIndex.from_product([text, range(0, DIMENSIONS)], names=['column', 'dimension'])
        text_embedding_stats = pd.DataFrame(columns=index)
        for column in text:
            base1_ps = self.text_embedding_ks_tests(column, self.data1_p1, self.data1_p2)
            base2_ps = self.text_embedding_ks_tests(column, self.data2_p1, self.data2_p2)
            ps = self.text_embedding_ks_tests(column, self.data1, self.data2)
            for d in range(0, DIMENSIONS):
                text_embedding_stats[(column, d)] = [base1_ps[d], base2_ps[d], ps[d]]
        return text_embedding_stats

    def analyze_text_metadata(self, text, text_metadata_types):
        index = pd.MultiIndex.from_product([text, text_metadata_types], names=['column', 'metadata'])
        text_meta_stats = pd.DataFrame(columns=index)
        for column in text:
            print('cross val 1')
            base1_ps = self.text_metadata_tests(column, self.data1_p1, self.data1_p2, text_metadata_types)
            print('cross val 2')
            base2_ps = self.text_metadata_tests(column, self.data2_p1, self.data2_p2, text_metadata_types)
            print('real test')
            ps = self.text_metadata_tests(column, self.data1, self.data2, text_metadata_types)
            for metadata_type in text_metadata_types:
                text_meta_stats[(column, metadata_type)] = [base1_ps[metadata_type], base2_ps[metadata_type],
                                                            ps[metadata_type]]
        return text_meta_stats

    def perform_stat_test(self, columns=[], significance=0.01, analyze_text_embedding=False,
                          analyze_text_metadata=False, text_metadata_types=None):

        if text_metadata_types is None:
            text_metadata_types = ['num_chars', 'num_words', 'distinct_words', 'language', 'lang_ambiguity']
        if not columns:
            columns = list(self.data1.columns)

        non_unique_columns = [column for column in columns if not is_unique(column, self.data1, self.data2)]

        numerical = [column for column in non_unique_columns if self.data1[column].dtype in [np.int64, np.float64]]
        categorical = [column for column in non_unique_columns if column not in numerical
                       and has_same_value_set(column, self.data1, self.data2)]  # TODO: decide differently
        text = [column for column in columns if self.data1[column].dtype not in [np.int64, np.float64]
                # NOTE: this differs from not being in numerical, because unique cols included
                and column not in categorical
                and has_reasonable_length(column, self.data1, self.data2)]
        strings = [column for column in columns if column not in numerical
                   and column not in categorical
                   and column not in text]

        num_stats = self.analyze_numerical_columns(numerical)

        cat_stats = self.analyze_categorical_columns(categorical)

        text_embedding_stats = self.analyze_text_embedding(text) if analyze_text_embedding else pd.DataFrame()

        text_meta_stats = self.analyze_text_metadata(text, text_metadata_types) if analyze_text_metadata \
            else pd.DataFrame()

        result = StatResult(num_stats, cat_stats, text_embedding_stats, text_meta_stats, significance=significance)
        return result

    def run(self, columns=[], analyze_text=False, text_metadata_types=None):
        return self.perform_stat_test(self.data1, self.data2, columns=columns,
                                      analyze_text_embedding=analyze_text, analyze_text_metadata=analyze_text,
                                      text_metadata_types=text_metadata_types)

    @staticmethod
    def text_embedding_ks_tests(column, part1, part2):
        pvalues = []
        clean1 = part1[column].dropna()
        clean2 = part2[column].dropna()
        vec_df = get_column_embedding(clean1, clean2)
        for d in vec_df.columns:
            dim_data1 = vec_df.loc[:len(clean1), d]
            dim_data2 = vec_df.loc[len(clean1):, d]
            p = ks_test_series(dim_data1, dim_data2)
            pvalues.append(p)
        return pvalues

    @staticmethod
    def text_metadata_tests(column, part1, part2, text_metadata_types):
        pvalues = {}
        clean1 = part1[column].dropna()
        clean2 = part2[column].dropna()
        metadata1 = pd.DataFrame()
        metadata2 = pd.DataFrame()
        for metadata_type in text_metadata_types:
            metadata1[metadata_type] = [text_md.md_functions(metadata_type)(text) for text in clean1]
            metadata2[metadata_type] = [text_md.md_functions(metadata_type)(text) for text in clean2]
            if metadata_type in ['num_chars', 'ratio_upper', 'num_words', 'distinct_words', 'num_parts',
                                 'lang_ambiguity']:
                pvalues[metadata_type] = ks_test(metadata_type, metadata1, metadata2)
            elif metadata_type in ['language', 'category']:
                pvalues[metadata_type] = chi_test(metadata_type, metadata1, metadata2)
            else:
                raise ValueError('Unknown metadata type provided')
        return pvalues


d1 = pd.read_csv('~/Documents/Shiftsets/real world/titanicB_p1.csv', sep='|')
d2 = pd.read_csv('~/Documents/Shiftsets/real world/titanicB_p2.csv', sep='|')
res = StatisticalTestAnalyzer(d1, d2).run(analyze_text=False)
print(res.significant_columns())
