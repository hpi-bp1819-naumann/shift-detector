from copy import deepcopy

from shift_detector.precalculations.low_cardinality_precalculation import LowCardinalityPrecalculation
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.precalculations.text_metadata import TextMetadata
from shift_detector.utils.column_management import ColumnType


class DQMetricsPrecalculation(Precalculation):

    def __init__(self, text_metadata=False):
        self.text_metadata = text_metadata

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    def process(self, store):
        if self.text_metadata:
            df1_metadata, df2_metadata = store[TextMetadata()]
            metadata_comparison = self.compare_numerical_columns(df1_metadata, df2_metadata)
        else:
            metadata_comparison = None

        df1_numerical, df2_numerical = store[ColumnType.numerical]
        df1_categorical, df2_categorical, _ = store[LowCardinalityPrecalculation()]

        numerical_comparison = self.compare_numerical_columns(df1_numerical, df2_numerical)
        categorical_comparison = self.compare_categorical_columns(df1_categorical, df2_categorical)
        attribute_val_comparison = self.compare_categorical_attribute_vals(df1_categorical, df2_categorical)
        combined_comparisons = {'attribute_val_comparison': attribute_val_comparison,
                                'numerical_comparison': numerical_comparison, 'categorical_comparison':
                                categorical_comparison, 'metadata_comparison': metadata_comparison}
        return combined_comparisons

    @staticmethod
    def compare_numerical_columns(df1, df2):
        numerical_comparison = dict()
        empty_metrics_dict = {'mean': {}, 'median': {}, 'value_range': {}, 'quartile_1': {}, 'quartile_3': {},
                              'uniqueness': {}, 'completeness': {}, 'std': {}}

        for df_name, df in [('df1', df1), ('df2', df2)]:
            for column in df.columns:
                if df_name == 'df1':
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)
                elif not numerical_comparison.get(column):
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)

                numerical_comparison[column]['value_range'][df_name] = float(df[column].max()) - float(df[column].min())
                numerical_comparison[column]['quartile_1'][df_name] = df[column].quantile(.25)
                numerical_comparison[column]['quartile_3'][df_name] = df[column].quantile(.75)

                numerical_comparison[column]['median'][df_name] = df[column].median()
                numerical_comparison[column]['mean'][df_name] = df[column].mean()

                column_droppedna = df[column].dropna()
                numerical_comparison[column]['std'][df_name] = column_droppedna.std()
                numerical_comparison[column]['completeness'][df_name] = len(column_droppedna) / len(df[column])
                numerical_comparison[column]['uniqueness'][df_name] = \
                    len(df.groupby(column).filter(lambda x: len(x) == 1)) / len(column_droppedna)
        return numerical_comparison

    @staticmethod
    def compare_categorical_columns(df1, df2):
        categorical_comparison = dict()
        empty_metrics_dict = {'num_distinct': {}, 'completeness': {}, 'uniqueness': {}}

        for df_name, df in [('df1', df1), ('df2', df2)]:
            for column in df.columns:
                if df_name == 'df1':
                    categorical_comparison[column] = deepcopy(empty_metrics_dict)
                elif not categorical_comparison.get(column):
                    categorical_comparison[column] = deepcopy(empty_metrics_dict)

                column_droppedna = df[column].dropna()
                categorical_comparison[column]['num_distinct'][df_name] = column_droppedna.nunique()
                categorical_comparison[column]['completeness'][df_name] = len(column_droppedna) / len(df[column])
                categorical_comparison[column]['uniqueness'][df_name] = \
                    len(df.groupby(column).filter(lambda x: len(x) == 1)) / len(column_droppedna)
        return categorical_comparison

    @staticmethod
    def compare_categorical_attribute_vals(df1, df2):
        category_comparison = {}

        for column in list(df1.columns):
            category_comparison[column] = {}
            attribute_ratios_df1 = df1[column].value_counts(normalize=True).to_dict()

            for key, value in attribute_ratios_df1.items():
                category_comparison[column][key] = {'df1': value, 'df2': 0}

            attribute_ratios_df2 = df2[column].value_counts(normalize=True).to_dict()
            for key, value in attribute_ratios_df2.items():
                if category_comparison[column].get(key):
                    category_comparison[column][key]['df2'] = value
                else:
                    category_comparison[column][key] = {'df1': 0, 'df2': value}

        return category_comparison
