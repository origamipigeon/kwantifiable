import symbols
from quote_data import get_symbol, save_symbol, update_all_symbols, clear, clear_and_get_full_data
import logging
import pandas as pd
from sklearn import preprocessing
import numpy as np
import keras
import tensorflow as tf
import math
from pprint import pprint

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from tensorflow import set_random_seed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

log = logging.getLogger('kwantifiable')

np.random.seed(42)
tf.compat.v1.set_random_seed(42)

class value_predictor:

    def __init__(self, value_type, epochs, historical_points):
        self.value_type = value_type
        self.epochs = epochs
        self.historical_values = historical_points

    def train(self):
        historical_values = self.historical_values
        for symbol in symbols.symbols_list:
            log.info('Loading {}'.format(symbol))
            symbol_data = get_symbol(symbol, 'daily_adj')
            # symbol data is open, high, low, adj close, volume

            # first row is most recent, so reverse data
            symbol_data = symbol_data[::-1]

            data_normaliser = preprocessing.MinMaxScaler()
            data_normalised = data_normaliser.fit_transform(symbol_data)

            # get 'batches' of data, each containing historical_values number of entries
            # skip first row as we don't know the next day value for it
            ohlcv_histories_normalised = np.array([data_normalised[i+1:i + historical_values + 1].copy() for i in range(len(data_normalised) - historical_values)])

            # col 3 = adj close, col 0 = open
            if self.value_type == 'open':
                col = 0
            elif self.value_type == 'close':
                col = 3
            else:
                exit()

            next_day_values_normalised = np.array([data_normalised[:, col][i + historical_values].copy() for i in range(len(data_normalised) - historical_values)])
            next_day_values_normalised = np.expand_dims(next_day_values_normalised, -1)

            # get the values for the day after the last day for each historical batch
            next_day_values = np.array([symbol_data[:, 3][i + historical_values].copy() for i in range(len(symbol_data) - historical_values)])
            next_day_values = np.expand_dims(next_day_values, -1)

            y_normaliser = preprocessing.MinMaxScaler()
            y_normaliser.fit(next_day_values)

            technical_indicators = []
            # todo: other indicators?
            for his in ohlcv_histories_normalised:
                sma = np.mean(his[:, 3])
                technical_indicators.append(np.array([sma]))

            technical_indicators = np.array(technical_indicators)

            tech_ind_scaler = preprocessing.MinMaxScaler()
            technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
            
            assert ohlcv_histories_normalised.shape[0] == next_day_values_normalised.shape[0] == technical_indicators_normalised.shape[0]

            test_split = 0.9

            n = int(ohlcv_histories_normalised.shape[0] * test_split)

            ohlcv_train = ohlcv_histories_normalised[:n]
            tech_ind_train = technical_indicators_normalised[:n]
            y_train = next_day_values_normalised[:n]

            ohlcv_test = ohlcv_histories_normalised[n:]
            tech_ind_test = technical_indicators_normalised[n:]
            y_test = next_day_values_normalised[n:]

            unscaled_y_test = next_day_values[n:]

            lstm_input = Input(shape=(historical_values, 5), name='lstm_input')
            dense_input = Input(shape=(technical_indicators_normalised.shape[1],), name='tech_input')

            # the first branch operates on the first input
            x = LSTM(historical_values, name='lstm_0')(lstm_input)
            x = Dropout(0.2, name='lstm_dropout_0')(x)
            lstm_branch = Model(inputs=lstm_input, outputs=x)

            # the second branch opreates on the second input
            y = Dense(20, name='tech_dense_0')(dense_input)
            y = Activation("relu", name='tech_relu_0')(y)
            y = Dropout(0.2, name='tech_dropout_0')(y)
            technical_indicators_branch = Model(inputs=dense_input, outputs=y)

            # combine the output of the two branches
            combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

            z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
            z = Dense(1, activation="linear", name='dense_out')(z)

            # our model will accept the inputs of the two branches and
            # then output a single value
            log.info('Training {} for {}'.format(self.value_type, symbol))
            model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
            adam = optimizers.Adam(lr=0.0005)
            model.compile(optimizer=adam, loss='mse')

            filepath = "models/{}_{}_model.h5".format(symbol, self.value_type)
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=self.epochs, shuffle=True, validation_split=0.1, callbacks=callbacks_list)

            start = 0
            end = -1

            y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
            y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
            assert unscaled_y_test.shape == y_test_predicted.shape
            real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
            scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
            rmse = math.sqrt(scaled_mse)

            print("RMSE for {}: {}".format(symbol, rmse))

            #model.save(filepath)

            #Begin Graph

            plt.gcf().set_size_inches(22, 15, forward=True)

            real = plt.plot(unscaled_y_test[start:end], label='real')
            pred = plt.plot(y_test_predicted[start:end], label='predicted')

            path = Path("graphs/")
            path.mkdir(parents=True, exist_ok=True)

            plt.legend(['Real', 'Predicted'])

            #plt.show()
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.title("{} {} test prediction (RMSE {})".format(symbol, self.value_type, rmse))
            plt.savefig("graphs/{}_{}_test.png".format(symbol, self.value_type))
            plt.close()

    def predict(self):
        historical_values = self.historical_values
        results = {}
        for symbol in symbols.symbols_list:
            #log.info('Loading {}'.format(symbol))
            symbol_data = get_symbol(symbol, 'daily_adj')
            # symbol data is open, high, low, adj close, volume

            # first row is most recent, so reverse data
            symbol_data = symbol_data[::-1]

            data_normaliser = preprocessing.MinMaxScaler()
            data_normalised = data_normaliser.fit_transform(symbol_data)
        
            # predict next day
            model = None
            try:
                model = load_model("models/{}_{}_model.h5".format(symbol, self.value_type))
            except Exception as e:
                log.error("Could not load models/{}_{}_model.h5".format(symbol, self.value_type))
                print(e)
                continue

            # get 'batches' of data, each containing historical_values number of entries
            # skip first row as we don't know the next day value for it
            ohlcv_histories_normalised = np.array([data_normalised[i+1:i + historical_values + 1].copy() for i in range(len(data_normalised) - historical_values)])
            # col 3 = adj close, col 1 = open
            if self.value_type == 'open':
                col = 0
            elif self.value_type == 'close':
                col = 3
            else:
                exit()

            next_day_values_normalised = np.array([data_normalised[:, col][i + historical_values].copy() for i in range(len(data_normalised) - historical_values)])
            next_day_values_normalised = np.expand_dims(next_day_values_normalised, -1)

            # get the values for the day after the last day for each historical batch
            next_day_values = np.array([symbol_data[:, 3][i + historical_values].copy() for i in range(len(symbol_data) - historical_values)])
            next_day_values = np.expand_dims(next_day_values, -1)

            y_normaliser = preprocessing.MinMaxScaler()
            y_normaliser.fit(next_day_values)

            technical_indicators = []
            # todo: other indicators?
            for his in ohlcv_histories_normalised:
                sma = np.mean(his[:, 3])
                technical_indicators.append(np.array([sma]))

            technical_indicators = np.array(technical_indicators)

            tech_ind_scaler = preprocessing.MinMaxScaler()
            technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
            
            assert ohlcv_histories_normalised.shape[0] == next_day_values_normalised.shape[0] == technical_indicators_normalised.shape[0]

            # get last 6 batches
            # first 5 will be the batches we have a known close val for, last wil be the batch to predict
            #ohlcv_histories_predict = np.array([data_normalised[-historical_values + i:].copy() for i in range(-5, 1, 1)])
            ohlcv_histories_predict = np.array([data_normalised[i:i + historical_values].copy() for i in range(len(data_normalised) - historical_values - 5, len(data_normalised) - historical_values + 1, 1)])
            #ohlcv_histories_real = np.array([data_normalised[i:i + historical_values].copy() for i in range(len(data_normalised) - historical_values - 1, len(data_normalised) - historical_values + 1, 1)])
            #print (ohlcv_histories_predict)
                        
            technical_indicators_predict = []
            for his in ohlcv_histories_predict:
                sma = np.mean(his[:, 3])
                technical_indicators_predict.append(np.array([sma]))
                       
            # add min, max from SMA
            technical_indicators_predict.append(np.array([technical_indicators.min()]))
            technical_indicators_predict.append(np.array([technical_indicators.max()]))
            technical_indicators_predict = np.array(technical_indicators_predict)
            tech_ind_scaler2 = preprocessing.MinMaxScaler()
            technical_indicators_predict_normalised = tech_ind_scaler2.fit_transform(technical_indicators_predict.reshape(8, -1))
            #print(technical_indicators_predict_normalised)
            # using index 1 as index 0 contains the last prediction for a known correct last value
            technical_indicators_predict_normalised = technical_indicators_predict_normalised[0:6]
            last_value = next_day_values[-1]          
            y_predicted = model.predict([ohlcv_histories_predict, technical_indicators_predict_normalised])
            y_predicted = y_normaliser.inverse_transform(y_predicted)

            # now have last 5 predictions for a line and a 6th 
            plt.gcf().set_size_inches(22, 15, forward=True)
            #print(y_predicted[-2:])
            #print(next_day_values[-5:])
            
            start = 0
            end = -1

            real_extrapolated = y_predicted[-2:].copy()


            real = plt.plot(next_day_values[-5:], label='real')
            real_ex = plt.plot(real_extrapolated, label='real extrapolated')
            pred_back = plt.plot(y_predicted[start:-1], label='predicted real')
            pred =  plt.plot(y_predicted[-2:], label='predicted')

            path = Path("graphs/")
            path.mkdir(parents=True, exist_ok=True)

            plt.legend(['Real', 'Real Extrapolated', 'Predicted Previous', 'Predicted Future'])
            #plt.legend(['Real', 'Predicted'])

            #plt.show()
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.axvline(0, 0, 1)
            plt.title("{} {} prediction".format(symbol, self.value_type))
            plt.savefig("graphs/{}_{}_prediction.png".format(symbol, self.value_type))
            plt.close()

            y_predicted_val = y_predicted[-1:]
            diff = y_predicted[-1:][0][0] - y_predicted[-2:-1][0][0]
            pc = ((y_predicted[-1:][0][0] / y_predicted[-2:-1][0][0]) - 1) * 100
            
            if (diff > 0):
                print("Next value predicted for {0:}: UP {1:} (+{2:.3f}%)".format(symbol, diff, pc))
            else:
                print("Next value predicted for {0:}: DOWN {1:} ({2:.3f}%)".format(symbol, diff, pc)) #y_predicted[-1:][0][0]
            #results[symbol] = [y_predicted[0][0], last_value[0], (y_predicted-last_value)[0][0]]
            results[symbol] = [diff]
                 
        pprint("results: {}".format(results))

            
            

