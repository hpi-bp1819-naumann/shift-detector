import time

import pandas as pd
import numpy as np
from scipy import stats
from datawig.utils import random_split

from shift_detector.checks.BasicCheck import BasicCheck, CheckResult
from shift_detector.utils.Miscellaneous import print_progress_bar
from shift_detector.utils.TextEmbedding import get_column_embedding, DIMENSIONS
import shift_detector.utils.TextMetadataFunctions as TextMd
from shift_detector.utils.Visualization import plot_histogram


class StatResult(CheckResult):

    def print_report(self):
        print('Columns with equal distribution probability below significance level ', self.significance)
        print('Numerical: ', self.significant_num_columns())
        print('Categorical: ', self.significant_cat_columns())
        if self.has_embedding_stats:
            print('Text (Embedding): ', self.significant_embedding_columns())
        if self.has_text_meta_stats:
            sig_cols = self.significant_text_metadata_columns()
            print('Text (Metadata): ', sig_cols)
            for column in sig_cols:
                print('\tSignificant Metrics:')
                metrics = self.significant_text_meta_metrics(column)
                print('\t', column, ': ', metrics)

    def __init__(self, num_stats, cat_stats, text_embedding_stats, text_meta_stats, significance=0.01,
                 significant_dim_ratio=0.0):
        super().__init__()
        self.num = num_stats
        self.cat = cat_stats
        self.has_embedding_stats = True if len(text_embedding_stats) > 0 else False
        self.has_text_meta_stats = True if len(text_meta_stats) > 0 else False
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
        if not self.has_embedding_stats:
            return set()
        ratios = {}
        try:
            for column in self.embedding.columns.levels[0]:
                column_dims = pd.DataFrame.from_records(self.embedding[column], index=['base1', 'base2', 'test'])
                differing_dimensions = [dimension for dimension in column_dims.columns
                                        if self.is_significant(column_dims[dimension].loc['base1'],
                                                               column_dims[dimension].loc['base2'],
                                                               column_dims[dimension].loc['test'])]
                ratios[column] = len(differing_dimensions) / len(column_dims.columns)
            return set(column for column in self.embedding.columns.levels[0]
                       if ratios[column] > self.significant_dim_ratio)
        except Exception as e:
            print('Error: Failed to derive significant columns from text embedding stats. Reason:', e)
            return set()

    def significant_text_metadata_columns(self):
        if not self.has_text_meta_stats:
            return set()
        columns = set()
        #try:
        for column in self.text_meta.columns.levels[0]:
            columns_metadata = pd.DataFrame.from_records(self.text_meta[column], index=['base1', 'base2', 'test'])
            if any(self.is_significant(columns_metadata[md].loc['base1'],
                                       columns_metadata[md].loc['base2'],
                                       columns_metadata[md].loc['test']) for md in columns_metadata.columns):
                columns.add(column)
        return columns
        #except Exception as e:
        #    print('Error: Failed to derive significant columns from text metadata stats. Reason:', e)
        #    return set()

    def significant_columns(self):
        return set().union(*[self.significant_num_columns(), self.significant_cat_columns(),
                             self.significant_embedding_columns(), self.significant_text_metadata_columns()])

    def significant_text_meta_metrics(self, column):
        columns_metadata = pd.DataFrame.from_records(self.text_meta[column], index=['base1', 'base2', 'test'])
        return set(md for md in columns_metadata.columns if self.is_significant(columns_metadata[md].loc['base1'],
                                                                                columns_metadata[md].loc['base2'],
                                                                                columns_metadata[md].loc['test']))


def has_same_value_set(column, part1, part2):
    return not set(part1[column].unique()).symmetric_difference(set(part2[column].unique()))


def has_low_cardinality(column, part1, part2, max_cardinality=None, max_distinctness=None):
    if max_cardinality is not None and max_distinctness is not None:
        raise ValueError('Maximum cardinality and maximum distinctness provided. Provide only one')
    if max_cardinality is not None:
        return len(part1[column].unique()) <= max_cardinality and len(part2[column].unique()) <= max_cardinality
    elif max_distinctness is not None:
        part1_frac = len(part1[column].unique()) / len(part1[column])
        part2_frac = len(part2[column].unique()) / len(part2[column])
        return part1_frac <= max_distinctness and part2_frac <= max_distinctness
    else:
        raise ValueError('Maximum cardinality and maximum distinctness were both None')


def is_unique(column, part1, part2):
    return len(part1[column].dropna().unique()) == len(part1[column].dropna()) and len(part2[column].dropna().unique()) == len(part2[column].dropna())


def has_reasonable_length(column, part1, part2, min_length=2):
    data = pd.concat([part1[column], part2[column]]).dropna()
    return np.median([len(entry.split(maxsplit=2)) for entry in data]) > min_length


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


class StatisticalTestCheck(BasicCheck):

    def __init__(self, df1, df2):
        BasicCheck.__init__(self, df1, df2)
        self.df1_part1, self.df1_part2 = random_split(self.df1, [0.5, 0.5])
        self.df2_part1, self.df2_part2 = random_split(self.df2, [0.5, 0.5])
        self.sample_size = min(len(self.df1), len(self.df2))

    def analyze_categorical_columns(self, categorical):
        cat_stats = pd.DataFrame()
        for column in categorical:
            base1_p = chi_test(column, self.df1_part1, self.df1_part2)
            base2_p = chi_test(column, self.df2_part1, self.df2_part2)
            p = chi_test(column, self.df1.sample(self.sample_size), self.df2.sample(self.sample_size))
            cat_stats[column] = [base1_p, base2_p, p]
            cat_stats.index = ['base1', 'base2', 'test']
        return cat_stats

    def analyze_numerical_columns(self, numerical):
        num_stats = pd.DataFrame()
        for column in numerical:
            base1_p = ks_test(column, self.df1_part1, self.df1_part2)
            base2_p = ks_test(column, self.df2_part1, self.df2_part2)
            p = ks_test(column, self.df1.sample(self.sample_size), self.df2.sample(self.sample_size))
            num_stats[column] = [base1_p, base2_p, p]
            num_stats.index = ['base1', 'base2', 'test']
        return num_stats

    def analyze_text_embedding(self, text):
        index = pd.MultiIndex.from_product([text, range(0, DIMENSIONS)], names=['column', 'dimension'])
        text_embedding_stats = pd.DataFrame(columns=index)
        for column in text:
            base1_ps = self.text_embedding_ks_tests(column, self.df1_part1, self.df1_part2)
            base2_ps = self.text_embedding_ks_tests(column, self.df2_part1, self.df2_part2)
            ps = self.text_embedding_ks_tests(column, self.df1.sample(self.sample_size), self.df2.sample(self.sample_size))
            for d in range(0, DIMENSIONS):
                text_embedding_stats[(column, d)] = [base1_ps[d], base2_ps[d], ps[d]]
        return text_embedding_stats

    def analyze_text_metadata(self, text, text_metadata_types):
        index = pd.MultiIndex.from_product([text, text_metadata_types], names=['column', 'metadata'])
        text_meta_stats = pd.DataFrame(columns=index)
        for column in text:
            print('cross val 1')
            base1_ps = self.text_metadata_tests(column, self.df1_part1, self.df1_part2, text_metadata_types)
            print('cross val 2')
            base2_ps = self.text_metadata_tests(column, self.df2_part1, self.df2_part2, text_metadata_types)
            print('real test')
            ps = self.text_metadata_tests(column, self.df1.sample(self.sample_size), self.df2.sample(self.sample_size), text_metadata_types)
            for metadata_type in text_metadata_types:
                text_meta_stats[(column, metadata_type)] = [base1_ps[metadata_type], base2_ps[metadata_type],
                                                            ps[metadata_type]]
        return text_meta_stats

    def perform_stat_test(self, columns=[], significance=0.01, analyze_text_embedding=False,
                          analyze_text_metadata=False, text_metadata_types=None):

        if text_metadata_types is None:
            text_metadata_types = ['num_chars', 'num_words', 'distinct_words']
        if not columns:
            columns = list(self.df1.columns)

        non_unique_columns = [column for column in columns if not is_unique(column, self.df1, self.df2)]

        numerical = [column for column in non_unique_columns if self.df1[column].dtype in [np.int64, np.float64]
                     and not has_low_cardinality(column, self.df1, self.df2, max_distinctness=0.005)]
        start_time = time.time()
        categorical = [column for column in non_unique_columns if column not in numerical
                       and has_low_cardinality(column, self.df1, self.df2, max_distinctness=0.1)]  # TODO: determine threshold
        end_time = time.time()
        print('Categorical time consumed: ', end_time-start_time)
        start_time = time.time()
        text = [column for column in columns if self.df1[column].dtype not in [np.int64, np.float64]
                and column not in categorical
                and has_reasonable_length(column, self.df1, self.df2)]
        end_time = time.time()
        print('Text time consumed: ', end_time-start_time)
        strings = [column for column in columns if column not in numerical
                   and column not in categorical
                   and column not in text]

        print('column types identified')
        print('Numerical: ', numerical)
        print('Categorical: ', categorical)
        print('Text: ', text)
        print('String: ', strings)

        num_stats = self.analyze_numerical_columns(numerical)

        cat_stats = self.analyze_categorical_columns(categorical)

        print('Start embedding analysis')
        start_time = time.time()
        text_embedding_stats = self.analyze_text_embedding(text) if analyze_text_embedding else pd.DataFrame()
        end_time = time.time()
        print('Embedding test time consumed: ', end_time-start_time)

        print('Start metadata analysis')
        start_time = time.time()
        text_meta_stats = self.analyze_text_metadata(text, text_metadata_types) if analyze_text_metadata \
            else pd.DataFrame()
        end_time = time.time()
        print('Metadata test time consumed: ', end_time - start_time)

        print('tests performed')

        result = StatResult(num_stats, cat_stats, text_embedding_stats, text_meta_stats, significance=significance)
        return result

    def run(self, columns=[], analyze_text_embedding=False, analyze_text_metadata=False, text_metadata_types=None):
        return self.perform_stat_test(columns=columns, analyze_text_embedding=analyze_text_embedding,
                                      analyze_text_metadata=analyze_text_metadata, text_metadata_types=text_metadata_types)

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
        print(column, ' - text metadata analysis:')
        for i, metadata_type in enumerate(text_metadata_types):
            print('computing metadata ', metadata_type, ':')
            mdtype_values = []
            for j, text in enumerate(clean1):
                mdtype_values.append(TextMd.md_functions(metadata_type)(text))
                print_progress_bar(j, len(clean1)-1, 50)
            metadata1[metadata_type] = mdtype_values
            #metadata1[metadata_type] = [TextMd.md_functions(metadata_type)(text) for text in clean1]
            mdtype_values = []
            for j, text in enumerate(clean2):
                mdtype_values.append(TextMd.md_functions(metadata_type)(text))
                print_progress_bar(j, len(clean2) - 1, 50)
            metadata2[metadata_type] = mdtype_values
            #metadata2[metadata_type] = [TextMd.md_functions(metadata_type)(text) for text in clean2]
            if metadata_type in ['num_chars', 'ratio_upper', 'num_words', 'distinct_words', 'num_parts',
                                 'lang_ambiguity']:
                pvalues[metadata_type] = ks_test(metadata_type, metadata1, metadata2)
            elif metadata_type in ['language', 'category']:
                pvalues[metadata_type] = chi_test(metadata_type, metadata1, metadata2)
            else:
                raise ValueError('Unknown metadata type provided')
            print_progress_bar(i, len(text_metadata_types)-1, 50)
        return pvalues


d1 = pd.read_csv('~/Documents/Shiftsets/real world/titanicB_p1.csv', sep='|')
d2 = pd.read_csv('~/Documents/Shiftsets/real world/titanicB_p2.csv', sep='|')
res = StatisticalTestCheck(d1, d2).run(analyze_text_metadata=True)
res.print_report()
