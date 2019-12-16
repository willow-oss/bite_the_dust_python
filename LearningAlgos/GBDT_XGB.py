# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:50:42 2019

@author: Haru
"""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class XGBReg():
    def __init__(self, params = {}, num_round = 1000):
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
    
    def train_and_predict(self, tr_X, va_X, tr_y, va_y, plot_bool = False):
        dtrain = xgb.DMatrix(tr_X, label = tr_y)
        dvalid = xgb.DMatrix(va_X, label = va_y)
        watchlist = [(dtrain, "train"), (dvalid, "eval")]
        evals_result = {}
        model = xgb.train(self.base_params, 
                          dtrain, 
                          self.num_round, 
                          early_stopping_rounds=100, 
                          evals_result=evals_result, 
                          evals = watchlist)
        
        pred_y = model.predict(dvalid)
        score = r2_score(va_y, pred_y)
        
        if plot_bool == True:
            train_metric = evals_result['train']['rmse']
            plt.plot(train_metric, label='train rmse')
            eval_metric = evals_result['eval']['rmse']
            plt.plot(eval_metric, label='eval rmse')
            plt.grid()
            plt.legend()
            plt.xlabel('rounds')
            plt.ylabel('rmse')
            plt.show()

        return model, score