import pandas as pd
from pandas import DataFrame

from shift_detector.utils.column_management import detect_column_types, ColumnType, \
    CATEGORICAL_MAX_RELATIVE_CARDINALITY, column_names
from shift_detector.utils.data_io import shared_column_names
from shift_detector.utils.errors import InsufficientDataError

MIN_DATA_SIZE = int(CATEGORICAL_MAX_RELATIVE_CARDINALITY * 100)


class Store:

    def __init__(self,
                 df1: DataFrame,
                 df2: DataFrame,
                 custom_column_types={}):
        self.verify_min_data_size(min([len(df1), len(df2)]))

        self.shared_columns = shared_column_names(df1, df2)
        self.df1 = df1[self.shared_columns]
        self.df2 = df2[self.shared_columns]

        if not isinstance(custom_column_types, dict):
            raise TypeError("column_types is not a dictionary."
                            "Received: {}".format(custom_column_types.__class__.__name__))

        if any([not column for column in custom_column_types.keys()]):
            raise TypeError("Not all keys of column_types are of type string."
                            "Received: {}".format(list(custom_column_types.keys())))

        if any([not column_type for column_type in custom_column_types.values()]):
            raise TypeError("Not all values of column_types are of type ColumnType."
                            "Received: {}".format(list(custom_column_types.values())))

        self.type_to_columns = detect_column_types(self.df1, self.df2, self.shared_columns)

        self.__apply_custom_column_types(custom_column_types)

        numerical_columns = self.column_names(ColumnType.numerical)
        self.df1[numerical_columns] = self.df1[numerical_columns].astype(float)
        self.df2[numerical_columns] = self.df2[numerical_columns].astype(float)

        print("Numerical columns: {}".format(", ".join(column_names(self.column_names(ColumnType.numerical)))))
        print("Categorical columns: {}".format(", ".join(column_names(self.column_names(ColumnType.categorical)))))
        print("Text columns: {}".format(", ".join(column_names(self.column_names(ColumnType.text)))))

        self.splitted_dfs = {column_type: (self.df1[columns], self.df2[columns])
                             for column_type, columns in self.type_to_columns.items()}
        self.preprocessings = {}

    def __getitem__(self, needed_preprocessing) -> DataFrame:
        if isinstance(needed_preprocessing, ColumnType):
            return self.splitted_dfs[needed_preprocessing]
        '''
        if not isinstance(needed_preprocessing, Precalculation):
            raise Exception("Needed Preprocessing must be of type Precalculation or ColumnType")
        '''
        if needed_preprocessing in self.preprocessings:
            print("- Use already executed {}".format(needed_preprocessing.__class__.__name__))
            return self.preprocessings[needed_preprocessing]

        print("- Executing {}".format(needed_preprocessing.__class__.__name__))
        preprocessing = needed_preprocessing.process(self)
        self.preprocessings[needed_preprocessing] = preprocessing
        return preprocessing

    def column_names(self, *column_types):
        if not column_types:
            return self.shared_columns

        if any([not isinstance(column_type, ColumnType) for column_type in column_types]):
            raise TypeError("column_types should be empty or of type ColumnType.")

        multi_columns = [self.type_to_columns[column_type] for column_type in column_types]
        flattened = {column for columns in multi_columns for column in columns}
        return list(flattened)

    @staticmethod
    def verify_min_data_size(size):
        if size < MIN_DATA_SIZE:
            raise InsufficientDataError(actual_size=size, expected_size=MIN_DATA_SIZE)

    def __apply_custom_column_types(self, custom_column_to_column_type):
        column_to_column_type = {}
        for column_type, columns in self.type_to_columns.items():
            # iterate over columns for column_types
            for column in columns:
                # apply custom column type
                if column in custom_column_to_column_type:
                    custom_column_type = custom_column_to_column_type[column]
                    self.__change_column_type(column, custom_column_type)
                    column_to_column_type[column] = custom_column_type
                else:
                    column_to_column_type[column] = column_type

        new_column_type_to_columns = {
            ColumnType.categorical: [],
            ColumnType.numerical: [],
            ColumnType.text: []
        }

        # revert back to old column structure
        for column, column_type in column_to_column_type.items():
            new_column_type_to_columns[column_type].append(column)

        self.type_to_columns = new_column_type_to_columns

    def __change_column_type(self, column, new_column_type):
        if new_column_type == ColumnType.numerical:
            try:
                self.df1[column] = pd.to_numeric(self.df1[column])
                self.df2[column] = pd.to_numeric(self.df2[column])
            except Exception as e:
                raise Exception("An error occurred during the conversion of column '{}' to the new column type '{}'. "
                                "{}".format(column, new_column_type.name, str(e)))

        elif new_column_type == ColumnType.categorical or new_column_type == ColumnType.text:
            self.df1[column] = self.df1[column].astype(str)
            self.df2[column] = self.df2[column].astype(str)
