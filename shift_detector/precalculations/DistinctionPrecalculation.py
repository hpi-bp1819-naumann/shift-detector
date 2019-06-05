from typing import Tuple

import numpy as np
import pandas as pd
from datawig import SimpleImputer
from datawig.iterators import ImputerIterDf
from datawig.utils import random_split

from shift_detector.precalculations.Precalculation import Precalculation
from datawig.utils import logger as datawig_logger
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class DistinctionPrecalculation(Precalculation):

    def __init__(self, columns, num_epochs=10):
        self.columns = columns
        self.output_column = '__shift_detector__dataset'
        self.output_path = 'tmp/basicChecks_params'
        self.num_epochs = num_epochs
        self.imputer = None

        datawig_logger.setLevel("ERROR")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return set(self.columns) == set(other.columns)

    def __hash__(self):
        items = sorted(self.columns) + [self.__class__]
        return hash(tuple(items))

    def process(self, store):
        """
        Runs check on provided columns
        :return: result of the check
        """
        self.imputer = SimpleImputer(
            input_columns=self.columns,
            output_column=self.output_column,
            output_path=self.output_path)

        df1 = store.df1[self.columns]
        df2 = store.df2[self.columns]

        df1_train, df1_test, df2_train, df2_test = self.prepare_dfs(df1, df2)

        train_df = pd.concat([df1_train, df2_train], ignore_index=True)
        test_df = pd.concat([df1_test, df2_test], ignore_index=True)

        self.imputer.fit(train_df, test_df, num_epochs=self.num_epochs)

        imputed = self.imputer.predict(test_df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        base_accuracy, permuted_accuracies = self.calculate_permuted_accuracies(df1_train, df2_train, self.columns)

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'base_accuracy': base_accuracy,
            'permuted_accuracies': permuted_accuracies
        }

    def label_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Set labels of the first dataframe to 'A' and those of the second dataframe to 'B'
        :param df1: first DataFrame
        :param df2: second DataFrame
        :return: tuple of labeled DataFrames
        """
        df1[self.output_column] = 'A'
        df2[self.output_column] = 'B'

        return df1, df2

    @staticmethod
    def sample_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample DataFrames to length of shorter DataFrame
        :param df1: first DataFrame
        :param df2: second DataFrame
        :return: tuple of sampled DataFrame
        """
        min_len = min(len(df1), len(df2))
        return df1.sample(n=min_len), df2.sample(n=min_len)

    def prepare_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a train and a test dataset, in which the number number of tuples
        that come from the first and the number of those from the second dataset are equal
        :param df1: first dataset
        :param df2: second dataset
        :return: tuple of train and test dataset
        """
        df1, df2 = self.label_dfs(df1, df2)
        df1_sampled, df2_sampled = self.sample_dfs(df1, df2)

        df1_train, df1_test = random_split(df1_sampled)
        df2_train, df2_test = random_split(df2_sampled)

        return df1_train, df1_test, df2_train, df2_test

    def get_accuracy(self, df):
        """
        Predict the label for df and calculate the accuracy.
        :param df: DataFrame
        :return: accuracy
        """
        imputed = self.imputer.predict(df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        return accuracy_score(y_true, y_pred)

    def permuted_accuracy(self, df1, df2, column):
        """
        Shuffle the column of both dfs and then switch it.
        Calculate the accuracy for the new DataFrame.
        Do this multiple times to receive a meaningful average accuracy.
        :param df1: first DataFrame
        :param df2: second DataFrame
        :param column: the column that will be permuted
        :return: averaged accuracy
        """
        accuracies = []
        df = pd.concat([df1, df2], ignore_index=True)

        for _ in range(5):
            col_rand1 = shuffle(df1[column])
            col_rand2 = shuffle(df2[column])

            col_rand = pd.concat([col_rand2, col_rand1], ignore_index=True)
            df[column] = col_rand

            accuracy = self.get_accuracy(df)
            accuracies.append(accuracy)

        return np.asarray(accuracies).mean()

    def calculate_permuted_accuracies(self, df1, df2, columns):
        """
        Calculate the base accuracy and the permuted accuracy for all columns.
        :param df1: first DataFrame
        :param df2: second DataFrame
        :param columns: columns to calculate the permuted accuracy for
        :return: base accuracy and the permuted accuracies as a dictionary from column to accuracy
        """
        df = pd.concat([df1, df2], ignore_index=True)
        base_accuracy = self.get_accuracy(df)

        permuted_accuracies = {col: self.permuted_accuracy(df1, df2, col) for col in columns}

        return base_accuracy, permuted_accuracies
