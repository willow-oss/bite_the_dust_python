# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:50:26 2019

@author: willow-oss
"""

import GPy
import numpy as np
from sklearn.metrics import r2_score
from learningalgos.algo_base import Algo_Base

class GPReg(Algo_Base):
    def __init__(self, params = {}):
        self.name = "Gaussian Process"
        pass

    def train(self, tr_x, tr_y):
        input_dim = tr_x.shape[1]
        output_dim = 1 if np.ndim(tr_y) == 1 else tr_y.shape[1]        
        self.kernel = GPy.kern.RBF(input_dim)
        tr_x = np.array(tr_x)
        tr_y = np.array(tr_y).reshape(-1, output_dim)
        self.model = GPy.models.GPRegression(tr_x, tr_y, kernel=self.kernel)
        self.model.optimize()
    
    def predict(self, va_x):
        pred_y_array, pred_y_var = self.model.predict(va_x)
        pred_y = []
        for s in pred_y_array:
            pred_y.extend(s)
        return pred_y, pred_y_var
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):
        self.train(tr_x, tr_y)
        va_x = np.array(va_x)
        pred_y, pred_y_var = self.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y, pred_y)        
        return self.model, score
    
