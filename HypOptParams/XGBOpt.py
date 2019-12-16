# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:21:13 2019

@author: Haru
"""
from optuna import trial, create_study
from functools import partial
from LearningAlgos import GBDT_XGB

class XGBRegOpt():
    
    def __init__(self):
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
        
    def objective(self, tr_X, va_X, tr_y, va_y, trial):
        self.params_space = {
                "booster" : trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),
                "max_depth" : trial.suggest_int('max_depth', 1, 9),
                "min_child_weight" : trial.suggest_int('min_child_weight', 1, 5),
                "subsample" : trial.suggest_discrete_uniform('subsample', 0.65, 0.95, 0.05),
                "colsample_bytree" : trial.suggest_discrete_uniform('colsample_bytree', 0.65, 0.95, 0.05)
                }
        
        self.base_params.update(self.params_space)
        model = GBDT_XGB.XGBReg(self.base_params)
        model, score = model.train_and_predict(tr_X, va_X, tr_y, va_y)
        return - score
    
    def fetch_best_params(self, tr_X, va_X, tr_y, va_y):
        f = partial(self.objective, tr_X, va_X, tr_y, va_y)
        study = create_study()
        study.optimize(f, n_trials = 100)
        print('params:', study.best_params)
        self.base_params.update(study.best_params)
        return self.base_params

