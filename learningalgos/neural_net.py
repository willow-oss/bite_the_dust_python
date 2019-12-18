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

class MLPReg():
    def __init__(self, params = {}):
        self.base_params = {
                "hidden_neurons" : 96,
                "hidden_layers" : 4,
                "batch_size" : 64,
                "input_dropout" : 0,
                "hidden_dropout" : 0,
                "learning_rate" : 0.001
                }
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, plot_learning_curve = False):
        input_dim = 1 if np.ndim(tr_x) == 1 else tr_x.shape[1]
        output_dim = 1 if np.ndim(tr_y) == 1 else tr_x.shape[1]
        input_dropout =  self.base_params["input_dropout"]
        hidden_layers = self.base_params["hidden_layers"]
        hidden_neurons = self.base_params["hidden_neurons"]
        hidden_dropout =  self.base_params["hidden_dropout"]
        batch_size = self.base_params["batch_size"]
        learning_rate = self.base_params["learning_rate"]
        
        
        model = Sequential()
        model.add(Dense(input_dim=input_dim, output_dim=hidden_neurons))
        model.add(Activation("relu"))
        model.add(Dropout(input_dropout))
        
        for i in range(hidden_layers):
            model.add(Dense(input_dim=hidden_neurons, output_dim=hidden_neurons))
            model.add(Activation("relu"))
            model.add(Dropout(hidden_dropout))
            
        model.add(Dense(input_dim=hidden_neurons, output_dim=output_dim))
        model.add(Activation("linear"))
        optimizer = Adam(lr=learning_rate)
        
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.fit(tr_x, tr_y, nb_epoch=1000, batch_size=batch_size)
        
        pred_y = model.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        
        return model, score

"""
class RNNReg():
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
    
    def transform_data(self, x, y, length_of_sequence):
        #under developing
        return x, y
        
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, plot_learning_curve = False):
        input_dim = 1 if np.ndim(tr_x) == 1 else tr_x.shape[1]
        output_dim = 1 if np.ndim(tr_y) == 1 else tr_x.shape[1]
        input_dropout =  self.base_params["input_dropout"]
        length_of_sequence = self.base_params["length_of_sequence"]
        hidden_neurons = self.base_params["hidden_neurons"]
        hidden_dropout =  self.base_params["hidden_dropout"]
        batch_size = self.base_params["batch_size"]
        learning_rate = self.base_params["learning_rate"]
        
        model = Sequential()
        
        model.add(SimpleRNN(hidden_neurons, 
                            input_shape = (length_of_sequence, input_dim),
                            return_sequences = False))
                    
        model.add(Dense(output_dim=output_dim))
        
        optimizer = Adam(lr=learning_rate)
        
        #reshaping
        tr_x = np.array(tr_x).reshape(len(tr_x), input_dim, 1)
        print(tr_x.shape)
        
        
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.fit(tr_x, tr_y, nb_epoch=1000, batch_size=batch_size, validation_split=0.1)
        
        pred_y = model.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        
        return model, score
"""