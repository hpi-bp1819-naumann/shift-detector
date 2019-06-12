from copy import deepcopy
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType


class SimplePrecalculation(Precalculation):

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    def process(self, store):
        df1_numerical, df2_numerical = store[ColumnType.numerical]
        df1_categorical, df2_categorical = store[ColumnType.categorical]

        numerical_comparison = self.compare_numerical_columns(df1_numerical, df2_numerical)
        categorical_comparison = self.compare_categorical_columns(df1_categorical, df2_categorical, store.columns)
        combined_comparisons = {'categorical_comparison': categorical_comparison,
                                'numerical_comparison': numerical_comparison}
        return combined_comparisons

    @staticmethod
    def compare_numerical_columns(df1, df2):
        numerical_comparison = dict()
        empty_metrics_dict = {'mean': {}, 'median': {}, 'min': {}, 'max': {}, 'quartile_1': {}, 'quartile_3': {},
                              'uniqueness': {}, 'num_distinct': {}, 'completeness': {}, 'std': {}}

        for df_name, df in [('df1', df1), ('df2', df2)]:
            for column in df.columns:
                if df_name == 'df1':
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)
                elif not numerical_comparison.get(column):
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)

                # TODO Later Vielleicht: verschnellerbar, in dem man alle Quantile gleichzeitig berechnet,
                #  also quantile([0, 0.25,  ... ]) oder Methoden selbst berechnet
                numerical_comparison[column]['min'][df_name] = df[column].min()
                numerical_comparison[column]['max'][df_name] = df[column].max()
                numerical_comparison[column]['quartile_1'][df_name] = df[column].quantile(.25)
                numerical_comparison[column]['quartile_3'][df_name] = df[column].quantile(.75)

                numerical_comparison[column]['median'][df_name] = df[column].median()
                numerical_comparison[column]['mean'][df_name] = df[column].mean()

                column_droppedna = df[column].dropna()
                numerical_comparison[column]['std'][df_name] = column_droppedna.std()

                numerical_comparison[column]['num_distinct'][df_name] = column_droppedna.nunique()

                numerical_comparison[column]['completeness'][df_name] = len(column_droppedna) / len(df1[column])

                numerical_comparison[column]['uniqueness'][df_name] = len(df.groupby(column)
                                                                    .filter(lambda x: len(x) == 1)) / \
                                                                    len(column_droppedna)
        return numerical_comparison

    @staticmethod
    def compare_categorical_columns(df1, df2, columns):
        category_comparison = {}

        for column in list(df1.columns):
            category_comparison[column] = {}
            attribute_ratios_df1 = df1[column].value_counts(normalize=True).to_dict()
            # category_comparison[column]['df1'] = {}
            # category_comparison[column]['df2'] = {}

            for key, value in attribute_ratios_df1.items():
                category_comparison[column][key] = {'df1': value}

            attribute_ratios_df2 = df2[column].value_counts(normalize=True).to_dict()
            for key, value in attribute_ratios_df2.items():
                if category_comparison[column].get(key):
                    category_comparison[column][key]['df2'] = value

        return category_comparison
