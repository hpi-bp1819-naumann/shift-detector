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


class LdaCheck(Check):

    def __init__(self, significance=0.1, n_topics=20, n_iter=10, lib='sklearn', random_state=0,
                 cols=None, trained_model=None, stop_words='english', max_features=None):
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
            self.cols = cols  # setting cols to None is equal to setting it to a list with all text columns

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
                                                                                     lib=self.lib,
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
                                                                                     lib=self.lib,
                                                                                     random_state=self.random_state,
                                                                                     cols=self.cols,
                                                                                     trained_model=self.trained_model,
                                                                                     stop_words=self.stop_words,
                                                                                     max_features=self.max_features)]
            all_dtms, all_vecs = (None, None)

        for col in col_names:
            count_topics1 = df1_embedded['topics ' + col].value_counts().sort_index()
            count_topics2 = df2_embedded['topics ' + col].value_counts().sort_index()

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
                      figures=self.column_figures(col_names, df1_embedded, df2_embedded, topic_words,
                                                  all_models, all_dtms, all_vecs, all_corpora, all_dicts))

    def column_figure(self, column, df1, df2, topic_words,
                      all_models, all_dtms, all_vecs, all_corpora, all_dicts):
        #self.paired_total_ratios_figure(column, df1, df2)
        self.word_cloud(column, topic_words, self.n_topics, self.lib)
        #self.py_ldavis(column, self.lib, all_models, all_dtms, all_vecs, all_corpora, all_dicts)

    def column_figures(self, significant_columns, df1, df2, topic_words,
                       all_models, all_dtms, all_vecs, all_corpora, all_dicts):
        plot_functions = []
        for column in significant_columns:
            plot_functions.append(lambda col=column: self.column_figure(col, df1, df2, topic_words,
                                                                        all_models, all_dtms, all_vecs,
                                                                        all_corpora, all_dicts))
        return plot_functions

    @staticmethod
    def paired_total_ratios_figure(column, df1, df2):
        value_counts = pd.concat([df1['topics ' + column].value_counts(), df2['topics ' + column].value_counts()],
                                 axis=1).sort_index()
        value_ratios = value_counts.fillna(0).apply(axis='columns',
                                                    func=lambda row: pd.Series([row.iloc[0] /
                                                                                len(df1['topics ' + column]),
                                                                                row.iloc[1] /
                                                                                len(df2['topics ' + column])],
                                                                               index=[str(column) + ' 1',
                                                                                      str(column) + ' 2']))
        axes = value_ratios.plot(kind='barh', fontsize='medium')
        axes.invert_yaxis()  # to match order of legend
        axes.set_title(str(column), fontsize='x-large')
        axes.set_xlabel('ratio', fontsize='medium')
        axes.set_ylabel('topics', fontsize='medium')
        plt.show()

    def word_cloud(self, column, topic_words, n_topics, lib):
        cols = [color for name, color in mcolors.XKCD_COLORS.items()]

        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          collocations=False,
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        j = int(np.ceil(n_topics / 2))

        fig, axes = plt.subplots(j, 2, figsize=(20, 10))

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            #ax.set_adjustable(adjustable='datalim')
            if lib == 'gensim':
                topics = dict(topic_words[column][i][1])
                cloud.generate_from_frequencies(topics, max_font_size=300)
            else:
                topics = ' '.join(set(topic_words[column][i]))
                cloud.generate_from_text(topics)

            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
            plt.gca().axis('off')

        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.show()

    @staticmethod
    def py_ldavis(column, lib, lda_models, dtm=None, vectorizer=None, corpus=None, dictionary=None):
        if lib == 'sklearn':
            print('prepare')
            vis_data = pyLDAvis.sklearn.prepare(lda_models[column], np.asmatrix(dtm[column]), vectorizer[column])
        else:
            print('prepare')
            vis_data = pyLDAvis.gensim.prepare(lda_models[column], corpus[column], dictionary[column])
        print('display')
        display(pyLDAvis.display(vis_data))


