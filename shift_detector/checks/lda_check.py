from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.utils.column_management import ColumnType
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
from IPython.display import display
import pandas as pd
import numpy as np
import warnings

from shift_detector.utils.visualization import LEGEND_1, LEGEND_2, plot_title


class LdaCheck(Check):

    def __init__(self, significance=0.1, n_topics=20, n_iter=10, lib='sklearn', random_state=0,
                 cols=None, trained_model=None, stop_words='english', max_features=None,
                 word_clouds=True, ldavis=True):
        """
        significance here is the difference between the percentages of each topic between both datasets,
        meaning a difference above 10% is significant
        """
        if not isinstance(significance, float):
            raise TypeError("Significance has to be a float. Received: {}".format(type(significance)))
        if not 0 < significance < 1:
            raise ValueError("Significance has to be between 0 and 1. Received: {}".format(significance))

        if cols:
            if isinstance(cols, list) and all(isinstance(col, str) for col in cols):
                self.cols = cols
            else:
                raise TypeError("Cols has to be list of strings . Column {} is of type {}".format(cols, type(cols)))
        else:
            # setting cols to None is equal to setting it to a list with all text columns
            self.cols = cols

        if trained_model:
            warnings.warn("Trained models are not trained again. Please make sure to only input the column(s) "
                          "that the model was trained on", UserWarning)
        self.trained_model = trained_model

        self.significance = significance
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.lib = lib
        self.random_state = random_state
        self.stop_words = stop_words
        self.max_features = max_features
        self.word_clouds = word_clouds
        self.ldavis = ldavis

    def run(self, store) -> Report:
        shifted_columns = set()
        explanation = {}

        if self.cols:
            for col in self.cols:
                if col not in store.column_names(ColumnType.text):
                    raise ValueError("Given column is not contained in detected text columns of the datasets: {}"
                                     .format(col))
            col_names = self.cols
        else:
            col_names = store.column_names(ColumnType.text)
            self.cols = list(col_names)
            col_names = self.cols

        if self.lib == 'sklearn':
            df1_embedded, df2_embedded, topic_words, all_models, all_dtms, all_vecs = store[LdaEmbedding(
                                                                                     n_topics=self.n_topics,
                                                                                     n_iter=self.n_iter,
                                                                                     lib='sklearn',
                                                                                     random_state=self.random_state,
                                                                                     cols=self.cols,
                                                                                     trained_model=self.trained_model,
                                                                                     stop_words=self.stop_words,
                                                                                     max_features=self.max_features)]
            all_corpora, all_dicts = (None, None)

        else:
            df1_embedded, df2_embedded, topic_words, all_models, all_corpora, all_dicts = store[LdaEmbedding(
                                                                                     n_topics=self.n_topics,
                                                                                     n_iter=self.n_iter,
                                                                                     lib='gensim',
                                                                                     random_state=self.random_state,
                                                                                     cols=self.cols,
                                                                                     trained_model=self.trained_model,
                                                                                     stop_words=self.stop_words,
                                                                                     max_features=self.max_features)]
            all_dtms, all_vecs = (None, None)

        for col in col_names:
            count_topics1 = df1_embedded['topics ' + col].value_counts()\
                            .reindex(np.arange(1, self.n_topics + 1))\
                            .sort_index()
            count_topics2 = df2_embedded['topics ' + col].value_counts()\
                            .reindex(np.arange(1, self.n_topics + 1))\
                            .sort_index()

            values1_ratio = [x / len(df1_embedded['topics ' + col]) for x in count_topics1.values]
            values2_ratio = [x / len(df2_embedded['topics ' + col]) for x in count_topics2.values]

            for i, (v1, v2) in enumerate(zip(values1_ratio, values2_ratio)):
                # number of rounded digits is 3 per default
                if abs(round(v1 - v2, 3)) >= self.significance:
                    shifted_columns.add(col)
                    explanation['Topic '+str(i+1)+' diff in column '+col] = round(v1 - v2, 3)

        return Report(check_name='LDA Check',
                      examined_columns=col_names,
                      shifted_columns=shifted_columns,
                      explanation=explanation,
                      figures=self.column_figures(shifted_columns, df1_embedded, df2_embedded, topic_words,
                                                  all_models, all_dtms, all_vecs, all_corpora, all_dicts,
                                                  self.word_clouds, self.ldavis))

    def column_figure(self, column, df1, df2, topic_words,
                      all_models, all_dtms, all_vecs, all_corpora, all_dicts,
                      word_clouds, ldavis):
        self.paired_total_ratios_figure(column, df1, df2, self.n_topics)
        if word_clouds:
            self.word_cloud(column, topic_words, self.n_topics)
        if ldavis:
            self.py_lda_vis(column, self.lib, all_models, all_dtms, all_vecs, all_corpora, all_dicts)

    def column_figures(self, shifted_columns, df1, df2, topic_words,
                       all_models, all_dtms, all_vecs, all_corpora, all_dicts,
                       word_clouds, ldavis):
        plot_functions = []
        for column in shifted_columns:
            plot_functions.append(lambda col=column: self.column_figure(col, df1, df2, topic_words,
                                                                        all_models, all_dtms, all_vecs,
                                                                        all_corpora, all_dicts,
                                                                        word_clouds, ldavis))
        return plot_functions

    @staticmethod
    def paired_total_ratios_figure(column, df1, df2, n_topics):
        count_topics1 = df1['topics ' + column].value_counts() \
            .reindex(np.arange(1, n_topics + 1)) \
            .sort_index()
        count_topics2 = df2['topics ' + column].value_counts() \
            .reindex(np.arange(1, n_topics + 1)) \
            .sort_index()
        value_counts = pd.concat([count_topics1, count_topics2], axis=1)

        value_ratios = value_counts.fillna(0).apply(axis='columns',
                                                    func=lambda row: pd.Series([row.iloc[0] /
                                                                                len(df1['topics ' + column]),
                                                                                row.iloc[1] /
                                                                                len(df2['topics ' + column])],
                                                                                index=['DS1',
                                                                                      'DS2']))
        axes = value_ratios.plot(kind='barh', fontsize='medium', figsize=(10, 2+np.ceil(n_topics/2)))
        axes.invert_yaxis()  # to match order of legend
        axes.set_title(plot_title(column), fontsize='x-large')
        axes.set_xlabel('ratio', fontsize='medium')
        axes.set_ylabel('topics', fontsize='medium')
        plt.show()

    @staticmethod
    def word_cloud(column, topic_words, n_topics):
        custom_XKCD_COLORS = mcolors.XKCD_COLORS
        # remove very light colors that are hard to see on white background
        custom_XKCD_COLORS.pop('xkcd:yellowish tan', None)
        custom_XKCD_COLORS.pop('xkcd:really light blue', None)

        cols = [color for name, color in custom_XKCD_COLORS.items()]

        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          collocations=False,
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        j = int(np.ceil(n_topics / 2))
        fig, axes = plt.subplots(j, 2, figsize=(10, 10+n_topics))

        for i, ax in enumerate(axes.flatten()):
            if n_topics % 2 == 1 and i == n_topics:
                fig.add_subplot(ax)
                ax.axis('off')
                break
            fig.add_subplot(ax)
            topics = dict(topic_words[column][i][1])
            cloud.generate_from_frequencies(topics, max_font_size=300)

            ax.imshow(cloud)
            ax.set_title('Topic ' + str(i+1), fontdict=dict(size=16))
            ax.axis('off')

        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.show()

    @staticmethod
    def py_lda_vis(column, lib, lda_models, dtm=None, vectorizer=None, corpus=None, dictionary=None):
        if lib == 'sklearn':
            vis_data = pyLDAvis.sklearn.prepare(lda_models[column],
                                                np.asmatrix(dtm[column]),
                                                vectorizer[column],
                                                sort_topics=False)
        else:
            vis_data = pyLDAvis.gensim.prepare(lda_models[column],
                                               corpus[column],
                                               dictionary[column],
                                               sort_topics=False)
        display(pyLDAvis.display(vis_data))


