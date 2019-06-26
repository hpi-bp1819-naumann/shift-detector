import logging as logger
import os
from typing import Tuple

import numpy as np
from gensim.models import FastText
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy.random import seed
from tensorflow import set_random_seed

from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.precalculations.text_embedding_precalculation import TextEmbeddingPrecalculation
from shift_detector.utils.column_management import ColumnType


class WordPredictionPrecalculation(Precalculation):

    def __init__(self, column, ft_window_size=5, ft_size=100, ft_workers=4, seed=None,
                 lstm_window=5, num_epochs_predictor=100, verbose=0,
                 output_path="wordPredictionCheck_model_checkpoints"):
        self.column = column
        self.ft_window_size = ft_window_size
        self.ft_size = ft_size
        self.seed = seed
        self.ft_workers = ft_workers
        self.lstm_window = lstm_window
        self.num_epochs_predictor = num_epochs_predictor
        self.verbose = verbose

        self.output_path = self.create_output_path(output_path)

        if not isinstance(self.column, str):
            raise ValueError('Column argument {} should be of type string. '
                             'Received {}.'.format(self.column, type(self.column)))

        if self.lstm_window < 1:
            raise ValueError('Expected argument lstm_window to be > 0. '
                             'Received {}.'.format(self.lstm_window))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.column == other.column \
            and self.ft_window_size == other.ft_window_size \
            and self.ft_size == other.ft_size \
            and self.seed == other.seed \
            and self.lstm_window == other.lstm_window \
            and self.num_epochs_predictor == other.num_epochs_predictor

    def __hash__(self):
        return hash((self.column, self.ft_window_size, self.ft_size,
                     self.seed, self.lstm_window, self.num_epochs_predictor))

    def process(self, store) -> Tuple[float, float]:

        if self.column not in store.column_names(ColumnType.text):
            raise ValueError('Column {} does not exist or is no textual column. '
                             'Please pass one of [{}] instead.'
                             .format(self.column, store.column_names(ColumnType.text)))

        ft_model = FastText(size=self.ft_size, window=self.ft_window_size, min_count=1,
                            workers=self.ft_workers, seed=self.seed)
        processed_df1, processed_df2 = store[TextEmbeddingPrecalculation(model=ft_model, agg=None)]

        seed(self.seed)
        set_random_seed(self.seed)

        df1_prediction_loss, df2_prediction_loss = self.get_prediction_losses(processed_df1,
                                                                              processed_df2,
                                                                              self.column)
        return df1_prediction_loss, df2_prediction_loss

    def get_prediction_losses(self, processed_df1, processed_df2, column) -> Tuple[float, float]:
        processed_df1 = processed_df1[[column]]
        processed_df2 = processed_df2[[column]]

        processed_df1_train, processed_df1_test = self.split(processed_df1)
        df1_train_x, df1_train_y = self.get_features_and_labels(processed_df1_train)
        df1_test_x, df1_test_y = self.get_features_and_labels(processed_df1_test)

        df2_test_x, df2_test_y = self.get_features_and_labels(processed_df2)

        # build and train model
        prediction_model = self.create_model()

        prediction_model.fit(df1_train_x, df1_train_y,
                             epochs=self.num_epochs_predictor, batch_size=512,
                             callbacks=self.create_callbacks(),
                             verbose=self.verbose,
                             validation_data=(df1_test_x, df1_test_y))

        # get prediction loss for both datasets
        df1_prediction_loss = prediction_model.evaluate(x=df1_test_x, y=df1_test_y,
                                                        verbose=self.verbose)
        df2_prediction_loss = prediction_model.evaluate(x=df2_test_x, y=df2_test_y,
                                                        verbose=self.verbose)

        return df1_prediction_loss, df2_prediction_loss

    def split(self, df, ratio=.8):
        msk = np.random.rand(len(df)) < ratio
        return df[msk], df[~msk]

    def get_features_and_labels(self, df):

        data = [np.array(row[0]) for row in df.values]

        train_data = []

        for i, row in enumerate(data):
            if row.shape[0] > self.lstm_window:
                for start_idx in range(row.shape[0] - self.lstm_window):
                    end_idx = start_idx + self.lstm_window + 1
                    train_data += [row[start_idx:end_idx, :]]
            else:
                logger.warning('Cannot use row {} for training. '
                               'Expected num words > lstm_window({}), but was {}'
                               .format(i, self.lstm_window, row.shape[0]))

        train_data = np.array(train_data)

        if train_data.shape == (0,):
            raise ValueError('Cannot execute Precalculation. '
                             'Text column does not contain any row with num words > lstm_window({})'
                             .format(self.lstm_window))

        features = train_data[:, :self.lstm_window, :]
        labels = train_data[:, self.lstm_window, :]

        return features, labels

    def create_output_path(self, output_path):

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def create_callbacks(self):
        callbacks = []

        callbacks += [ModelCheckpoint(self.output_path + '/model.h5', verbose=self.verbose,
                                      monitor='loss', save_best_only=True, mode='auto')]
        callbacks += [EarlyStopping(monitor='loss', patience=3, verbose=self.verbose,
                                    mode='auto', restore_best_weights=True)]

        return callbacks

    def create_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.lstm_window, self.ft_size)))
        model.add(Dense(self.ft_size))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
