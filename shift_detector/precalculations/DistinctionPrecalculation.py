from typing import Tuple

import numpy as np
import pandas as pd
from datawig import SimpleImputer
from datawig.iterators import ImputerIterDf
from datawig.utils import random_split

from shift_detector.precalculations.Precalculation import Precalculation


class DistinctionPrecalculation(Precalculation):

    def __init__(self, columns=[], num_epochs=10):
        self.columns = columns
        self.output_column = '__shift_detector__dataset'
        self.output_path = 'tmp/basicChecks_params'
        self.num_epochs = num_epochs
        self.imputer = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return set(self.columns) == set(other.columns)

    def __hash__(self):
        return hash(tuple(set([self.__class__]) | set(self.columns)))

    def process(self, store):
        """
        Runs check on provided columns
        :return: result of the check
        """
        input_columns = self.columns
        if not input_columns:
            input_columns = store.columns

        self.imputer = SimpleImputer(
            input_columns=input_columns,
            output_column=self.output_column,
            output_path=self.output_path)

        df1 = store.df1[input_columns]
        df2 = store.df2[input_columns]

        train_df, test_df = self.prepare_datasets(df1, df2)
        self.imputer.fit(train_df, test_df, num_epochs=self.num_epochs)

        imputed = self.imputer.predict(test_df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'relevant_columns': self.relevant_columns(input_columns, train_df)
        }

    def label_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Set labels of the first dataframe to 'A' and those of the second dataframe to 'B'
        :param df1: first dataset
        :param df2: second dataset
        :return: tuple of labeled datasets
        """
        df1[self.output_column] = 'A'
        df2[self.output_column] = 'B'

        return df1, df2

    def sample_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample datasets to length of shorter dataset
        :param df1: first dataset
        :param df2: second dataset
        :return: tuple of sampled datasets
        """
        min_len = min(len(df1), len(df2))
        return df1.sample(n=min_len), df2.sample(n=min_len)

    def prepare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a train and a test dataset, in which the number number of tuples
        that come from the first and the number of those from the second dataset are equal
        :param df1: first dataset
        :param df2: second dataset
        :return: tuple of train and test dataset
        """
        df1, df2 = self.label_datasets(df1, df2)
        df1_sampled, df2_sampled = self.sample_datasets(df1, df2)

        df1_train, df1_test = random_split(df1_sampled)
        df2_train, df2_test = random_split(df2_sampled)

        train_df = pd.concat([df1_train, df2_train])
        test_df = pd.concat([df1_test, df2_test])

        return train_df, test_df

    def get_loss(self, df):
        iterator = ImputerIterDf(
            data_frame=df,
            data_columns=self.imputer.imputer.data_encoders,
            label_columns=self.imputer.imputer.label_encoders,
            batch_size=self.imputer.imputer.batch_size
        )
        return self.imputer.imputer.module.predict(iterator)[0].asnumpy().mean()

    def loss_with_random_permutation(self, df, column):
        losses = []
        for _ in range(5):
            df_permutation = df.copy().sort_values(self.output_column, ascending=True)
            mid_row = int(df_permutation.size / 2)

            first_rand_permutation = np.random.permutation(df_permutation[column][:mid_row].values)
            second_rand_permutation = np.random.permutation(df_permutation[column][mid_row:].values)

            df_permutation[column][:len(second_rand_permutation)] = second_rand_permutation
            df_permutation[column][len(second_rand_permutation):] = first_rand_permutation

            losses += [self.get_loss(df_permutation)]
        return np.asarray(losses).mean()

    def relevant_columns(self, columns, test_df, thresh_percentage=.01):
        base_loss = self.get_loss(test_df)
        relevant_columns = []

        for col in columns:
            loss = self.loss_with_random_permutation(test_df, col)
            if loss > base_loss * (1 + thresh_percentage):
                relevant_columns += [col]

        return relevant_columns
