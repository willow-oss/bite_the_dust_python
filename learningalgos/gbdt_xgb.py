# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:50:42 2019

@author: willow-oss
"""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from learningalgos.algo_base import Algo_Base

class XGBReg(Algo_Base):
    def __init__(self, params = {}, num_round = 1000):
        self.name = "XGBoost"
        self.base_params = {"booster" : "gbtree",
          "objective" : "reg:squarederror",
          "eta" : 0.05,
          "gamma":0,
          "alpha":0.01,
          "lambda":1,
          "min_child_weight":1,
          "max_depth":5,
          "subsample":0.8,
          "colsample_bytree":0.8}
        self.num_round = num_round
        self.base_params.update(params)
    
    def predict(self, va_x):
        dvalid = xgb.DMatrix(va_x)
        return self.model.predict(dvalid)
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):
        dtrain = xgb.DMatrix(tr_x, label = tr_y)
        dvalid = xgb.DMatrix(va_x, label = va_y)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        evals_result = {}
        self.model = xgb.train(self.base_params, 
                          dtrain, 
                          self.num_round, 
                          early_stopping_rounds=100, 
                          evals_result=evals_result, 
                          evals = watchlist,
                          verbose_eval=False)
        
        pred_y = self.predict(va_x)
        score = r2_score(va_y, pred_y)
        
        if plot_learning_curve == True:
            train_metric = evals_result['train']['rmse']
            plt.plot(train_metric, label='train rmse')
            eval_metric = evals_result['eval']['rmse']
            plt.plot(eval_metric, label='eval rmse')
            plt.grid()
            plt.legend()
            plt.xlabel('rounds')
            plt.ylabel('rmse')
            plt.show()
        
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y, pred_y)
        return self.model, score
    
    def train(self, tr_x, tr_y):
        dtrain = xgb.DMatrix(tr_x, label = tr_y)
        evals_result = {}
        self.model = xgb.train(self.base_params, 
                  dtrain, 
                  self.num_round, 
                  early_stopping_rounds=100, 
                  evals_result=evals_result, 
                  verbose_eval=False)
        return self.model
    
