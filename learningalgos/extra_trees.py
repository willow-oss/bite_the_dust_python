# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:50:42 2019

@author: willow-oss
"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from learningalgos.algo_base import Algo_Base

class ETReg(Algo_Base):
    def __init__(self, params = {}):
        self.name = "Extra Trees"
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
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):
        self.model = ExtraTreesRegressor(**self.base_params)
        self.train(tr_x, tr_y)
        pred_y = self.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y, pred_y)
        return self.model, score
    


        