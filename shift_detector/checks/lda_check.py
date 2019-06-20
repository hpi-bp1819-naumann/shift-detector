from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.lda_embedding import LdaEmbedding
from shift_detector.utils.column_management import ColumnType
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import warnings



class LdaCheck(Check):

    def __init__(self, significance=10, n_topics=20, n_iter=10, lib='sklearn', random_state=0,
                 cols=None, trained_model=None, stop_words='english', max_features=None):
        """
        significance here is the difference between the percentages of each topic between both datasets,
        meaning a difference above 10% is significant
        """
        if not isinstance(significance, int):
            raise TypeError("Significance has to be an integer. Received: {}".format(type(significance)))
        if not significance > 0 or not significance < 100:
            raise ValueError("Significance has to be between 0% and 100%. Received: {}".format(significance))

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

        df1_embedded, df2_embedded, twd = store[LdaEmbedding(n_topics=self.n_topics, n_iter=self.n_iter, lib=self.lib,
                                                             random_state=self.random_state, cols=self.cols,
                                                             trained_model=self.trained_model, stop_words=self.stop_words,
                                                             max_features=self.max_features)]

        for col in col_names:
            count_topics1 = df1_embedded['topics ' + col].value_counts().sort_index()
            count_topics2 = df2_embedded['topics ' + col].value_counts().sort_index()



            #labels1_ordered, values1_ordered = zip(*sorted(count_topics1.items(), key=lambda kv: kv[0]))
            values1_percentage = [x * 100 / len(df1_embedded['topics ' + col]) for x in count_topics1.values]

            #labels2_ordered, values2_ordered = zip(*sorted(count_topics2.items(), key=lambda kv: kv[0]))
            values2_percentage = [x * 100 / len(df2_embedded['topics ' + col]) for x in count_topics2.values]

            print(values1_percentage)
            print(values2_percentage)

            for i, (v1, v2) in enumerate(zip(values1_percentage, values2_percentage)):
                print(abs(round(v1 - v2, 1)))
                # number of rounded digits is 1 per default
                if abs(round(v1 - v2, 1)) >= self.significance:
                    shifted_columns.add(col)
                    explanation['Topic '+str(i)+' diff in column '+col] = round(v1 - v2, 1)

        return Report(check_name='LDA Check',
                      examined_columns=col_names,
                      shifted_columns=shifted_columns,
                      explanation=explanation,
                      figures=self.column_figures(col_names, df1_embedded, df2_embedded))

    @staticmethod
    def paired_total_ratios_figure(column, df1, df2):
        value_counts = pd.concat([df1['topics ' + column].value_counts(), df2['topics ' + column].value_counts()],
                                 axis=1).sort_index()
        value_ratios = value_counts.fillna(0).apply(axis='columns',
                                                    func=lambda row: pd.Series([100 * row.iloc[0] /
                                                                                len(df1['topics ' + column]),
                                                                                100 * row.iloc[1] /
                                                                                len(df2['topics ' + column])],
                                                                               index=[str(column) + ' 1',
                                                                                      str(column) + ' 2']))
        print(value_ratios)
        axes = value_ratios.plot(kind='barh', fontsize='medium')
        axes.invert_yaxis()  # to match order of legend
        axes.set_title(str(column), fontsize='x-large')
        axes.set_xlabel('percentage', fontsize='medium')
        axes.set_ylabel('topics', fontsize='medium')
        plt.show()

    def column_figure(self, column, df1, df2, twd):
        self.paired_total_ratios_figure(column, df1, df2)
        self.wordcloud(column, twd)

    def column_figures(self, significant_columns, df1, df2, twd):
        plot_functions = []
        for column in significant_columns:
            plot_functions.append(lambda col=column: self.column_figure(col, df1, df2, twd))
        return plot_functions

    @staticmethod
    def wordcloud(column, twd):
        wordcloud = WordCloud(background_color='white').generate(' '.join(twd[column])) # add loop for topics
        plt.figure(figsize=(12, 12))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
