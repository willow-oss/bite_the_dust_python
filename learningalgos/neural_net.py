# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:50:26 2019

@author: willow-oss
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from learningalgos.algo_base import Algo_Base

class MLPReg(Algo_Base):
    def __init__(self, params = {}):
        
        self.base_params = {
                "hidden_neurons" : 96,
                "hidden_layers" : 4,
                "batch_size" : 64,
                "input_dropout" : 0,
                "hidden_dropout" : 0,
                "learning_rate" : 0.001
                }
    
    def train(self, tr_x, tr_y):
        input_dim = 1 if np.ndim(tr_x) == 1 else tr_x.shape[1]
        output_dim = 1 if np.ndim(tr_y) == 1 else tr_y.shape[1]
        input_dropout =  self.base_params["input_dropout"]
        hidden_layers = self.base_params["hidden_layers"]
        hidden_neurons = self.base_params["hidden_neurons"]
        hidden_dropout =  self.base_params["hidden_dropout"]
        batch_size = self.base_params["batch_size"]
        learning_rate = self.base_params["learning_rate"]
        
        self.model = Sequential()
        self.model.add(Dense(input_dim=input_dim, output_dim=hidden_neurons))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(input_dropout))
        
        for i in range(hidden_layers):
            self.model.add(Dense(input_dim=hidden_neurons, output_dim=hidden_neurons))
            self.model.add(Activation("relu"))
            self.model.add(Dropout(hidden_dropout))
            
        self.model.add(Dense(input_dim=hidden_neurons, output_dim=output_dim))
        self.model.add(Activation("linear"))
        optimizer = Adam(lr=learning_rate)
        
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)
        self.model.fit(tr_x, tr_y, nb_epoch=1000, batch_size=batch_size)

    def predict(self, va_x):
        pred_y_array = self.model.predict(va_x)
        pred_y = []
        for s in pred_y_array:
            pred_y.extend(s)        
        return pred_y
        
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):
        self.train(tr_x, tr_y)
        pred_y = self.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y, pred_y)   
        return self.model, score
        


class RNNReg(Algo_Base):
    def __init__(self, params = {}, model_type = "SimpleRNN"):
        self.base_params = {
                "hidden_neurons" : 96,
                "length_of_sequence": 10,
                "batch_size" : 64,
                "input_dropout" : 0,
                "hidden_dropout" : 0,
                "learning_rate" : 0.001
                }
        self.model_type = model_type
    
    def generate_data_for_learning(self, tr_x, tr_y):
        # DataFrame→array変換
        tr_x = tr_x.as_matrix()
        tr_y = tr_y.as_matrix()
        sequences = []
        target = []
    
        # 一グループごとに時系列データと正解データをセットしていく
        for i in range(0, tr_x.shape[0] - self.length_of_sequence):
            sequences.append(tr_x[i:i + self.length_of_sequence])
            target.append(tr_y[i + self.length_of_sequence - 1])    
        # 時系列データを成形
        X = np.array(sequences).reshape(len(sequences), self.length_of_sequence, self.input_dim)
        # 正解データを成形
        Y = np.array(target).reshape(len(sequences), self.output_dim)
        return X, Y        
    
    def train(self, tr_x, tr_y, plot_learning_curve = False):
        self.input_dim = 1 if np.ndim(tr_x) == 1 else tr_x.shape[1]
        self.output_dim = 1 if np.ndim(tr_y) == 1 else tr_x.shape[1]
        input_dropout =  self.base_params["input_dropout"]
        self.length_of_sequence = self.base_params["length_of_sequence"]
        hidden_neurons = self.base_params["hidden_neurons"]
        hidden_dropout =  self.base_params["hidden_dropout"]
        batch_size = self.base_params["batch_size"]
        learning_rate = self.base_params["learning_rate"]
        
        tr_x_RNN, tr_y_RNN = self.generate_data_for_learning(tr_x, tr_y)
        
        self.model = Sequential()
        
        self.model.add(SimpleRNN(hidden_neurons, 
                            input_shape = (self.length_of_sequence, self.input_dim),
                            return_sequences = False))
                    
        self.model.add(Dense(output_dim=self.output_dim))
        
        optimizer = Adam(lr=learning_rate)

                
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)
        self.model.fit(tr_x_RNN, tr_y_RNN, nb_epoch=1000, batch_size=batch_size, validation_split=0.1)
        
    def predict(self, va_x):
        va_x = va_x.as_matrix()
        sequences = []
        sequences.append(va_x[- self.length_of_sequence : -1])
        va_x_RNN = np.array(sequences).reshape(1, self.length_of_sequence, self.input_dim)
        pred_y = self.model.predict(va_x_RNN)[0]
        return pred_y

    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):

        self.train(tr_x, tr_y)
        va_x_RNN, va_y_RNN = self.generate_data_for_learning(va_x, va_y)
        pred_y_array = self.model.predict(va_x_RNN)
        pred_y = []
        va_y_result = []
        for s in pred_y_array:
            pred_y.extend(s)
        for s in va_y_RNN :
            va_y_result.append(s[0])
        score = r2_score(va_y_result, pred_y)
        print(va_y_result, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y_result, pred_y)
        return self.model, score    