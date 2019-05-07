from datawig import SimpleImputer
from datawig.iterators import ImputerIterDf
from datawig.utils import random_split
from datawig.utils import logger as dataWig_logger
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report
import numpy as np
import logging as logger
from shift_detector.analyzers.analyzer import Analyzer, AnalyzerResult


class BasicAnalyzerResult(AnalyzerResult):

    def __init__(self, result={}):
        AnalyzerResult.__init__(self, result)

    
    def get_columns_with_shift(self):
        return self.result['relevant_columns']


    def get_classification_report(self):
        return self.result['classification_report']


    def get_f1_score(self, className: str):
        return self.result['classification_report'][className]['f1-score']


class BasicAnalyzer(Analyzer):

    def __init__(self, data1: pd.DataFrame, data2: pd.DataFrame):

        Analyzer.__init__(self, data1, data2)

        self.output_column = '__shiftDetecor__dataset'
        self.output_path = 'tmp/basicAnalyzer_params'

        self.train_df = None
        self.test_df = None

        self.imputer = None


    def label_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Set labels of the first dataframe to 'A' and those of the second dataframe to 'B'

        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of labeled datasets

        """
        first_df[self.output_column] = 'A'
        second_df[self.output_column] = 'B'

        return first_df, second_df


    def sample_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Sample datasets to length of shorter dataset

        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of sampled datasets
        
        """
        min_len = min(len(first_df), len(second_df))
        return first_df.sample(n=min_len), second_df.sample(n=min_len)


    def prepare_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Create a train and a test dataset, in which the number number of tuples 
        that come from the first and the number of those from the second dataset are equal

        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of train and test dataset

        """
        first_df, second_df = self.label_datasets(first_df, second_df)
        first_df_sampled, second_df_sampled = self.sample_datasets(first_df, second_df)
        
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
        

    def run(self, columns=[]) -> BasicAnalyzerResult:
        """

        Runs analyzer on provided columns

        :param columns:
        :return: AnalyzerResult

        """

        self.imputer = SimpleImputer(
            input_columns=columns,
            output_column=self.output_column,
            output_path=self.output_path)

        self.train_df, self.test_df = self.prepare_datasets(self.data1, self.data2)
        self.imputer.fit(self.train_df, self.test_df)

        imputed = self.imputer.predict(self.test_df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        result = {}
        result['classification_report'] = classification_report(y_true, y_pred)
        result['relevant_columns'] = self.relevant_columns(columns)

        return BasicAnalyzerResult(result=result)

