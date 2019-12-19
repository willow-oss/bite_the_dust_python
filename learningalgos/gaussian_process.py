# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:50:26 2019

@author: willow-oss
"""

import GPy
import numpy as np
from sklearn.metrics import r2_score

class GPReg():
    def __init__(self, params = {}):
        pass
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y):
        input_dim = tr_x.shape[1]
        kernel = GPy.kern.RBF(input_dim)
        tr_x = np.array(tr_x)
        va_x = np.array(va_x)
        tr_y = np.array(tr_y).reshape(-1, 1)
        
        model = GPy.models.GPRegression(tr_x, tr_y, kernel=kernel)
        model.optimize()
        pred_y_mean, pred_y_var = model.predict(va_x)
        score = r2_score(va_y, pred_y_mean)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        
        return model, score