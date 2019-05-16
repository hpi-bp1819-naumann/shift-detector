from datawig import SimpleImputer
from datawig.iterators import ImputerIterDf
from datawig.utils import random_split
from datawig.utils import logger as dataWig_logger
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report
import numpy as np
import logging as logger
from shift_detector.checks.Check import Check, CheckResult


class BasicCheckResult(CheckResult):

    def __init__(self, result={}):
        CheckResult.__init__(self, result)

    def get_columns_with_shift(self):
        return self.result['relevant_columns']

    def get_classification_report(self):
        return self.result['classification_report']

    def get_f1_score(self, className: str):
        return self.result['classification_report'][className]['f1-score']


class BasicCheck(Check):

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):

        Check.__init__(self, df1, df2)

        self.output_column = '__shiftDetecor__dataset'
        self.output_path = 'tmp/basicChecks_params'

        self.train_df = None
        self.test_df = None

        self.imputer = None

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
        first_df_sampled, second_df_sampled = self.sample_datasets(df1, df2)
        
        first_df_train, first_df_test = random_split(first_df_sampled)
        second_df_train, second_df_test = random_split(second_df_sampled)
        
        train_df = pd.concat([first_df_train, second_df_train])
        test_df = pd.concat([first_df_test, second_df_test])

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

    def relevant_columns(self, columns, thresh_percentage=.01):

        base_loss = self.get_loss(self.test_df)
        relevant_columns = []

        for col in columns:
            loss = self.loss_with_random_permutation(self.test_df, col)
            if loss > base_loss * (1 + thresh_percentage):
                relevant_columns += [col]

        logger.info('The shifted columns are the following: {}'.format(relevant_columns))
        return relevant_columns

    def run(self, columns=[]) -> BasicCheckResult:
        """

        Runs check on provided columns

        :param columns:
        :return: CheckResult

        """

        self.imputer = SimpleImputer(
            input_columns=columns,
            output_column=self.output_column,
            output_path=self.output_path)

        self.train_df, self.test_df = self.prepare_datasets(self.df1, self.df2)
        self.imputer.fit(self.train_df, self.test_df)

        imputed = self.imputer.predict(self.test_df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        result = dict()
        result['classification_report'] = classification_report(y_true, y_pred)
        result['relevant_columns'] = self.relevant_columns(columns)

        return BasicCheckResult(result=result)

