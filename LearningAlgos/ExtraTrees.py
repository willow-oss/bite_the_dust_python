# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:50:42 2019

@author: Haru
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
    
    def train_and_predict(self, tr_X, va_X, tr_y, va_y, plot_bool = False):
        model = ExtraTreesRegressor(**self.base_params)
        model.fit(tr_X, tr_y)
        
        pred_y = model.predict(va_X)
        score = r2_score(va_y, pred_y)
        
        return model, score