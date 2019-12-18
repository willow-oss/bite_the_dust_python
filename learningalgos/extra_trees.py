# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:50:42 2019

@author: willow-oss
"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

class ETReg():
    def __init__(self, params = {}):
        self.base_params = {"n_estimators":1000,
                                   "criterion":'mae',
                                   "max_features":'auto',
                                   "max_depth":None,
                                   "bootstrap":True,
                                   "min_samples_split":4,
                                   "min_samples_leaf":1,
                                   "min_weight_fraction_leaf":0,
                                   "max_leaf_nodes":None}
        self.base_params.update(params)
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y):
        model = ExtraTreesRegressor(**self.base_params)
        model.fit(tr_x, tr_y)
        pred_y = model.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        
        return model, score