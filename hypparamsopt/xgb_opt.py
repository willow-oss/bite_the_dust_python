# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:21:13 2019

@author: willow-oss
"""
from optuna import trial, create_study
from functools import partial
from learningalgos import gbdt_xgb
from sklearn.model_selection import train_test_split

class XGBRegOpt():
    
    def __init__(self, params_space = {}, shuffle = True):
        self.base_params = {
                "booster" : "gbtree",
                "objective" : "reg:squarederror",
                "eta" : 0.05,
                "gamma":0,
                "alpha":0.01,
                "lambda":1,
                "min_child_weight":1,
                "max_depth":5,
                "subsample":0.8,
                "colsample_bytree":0.8}
        self.params_space = params_space
        self.shuffle = shuffle
        
    def _objective(self, x, y, trial):
        self.base_params_space = {
            "booster" : trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),
            "max_depth" : trial.suggest_int('max_depth', 1, 9),
            "min_child_weight" : trial.suggest_int('min_child_weight', 1, 5),
            "subsample" : trial.suggest_discrete_uniform('subsample', 0.65, 0.95, 0.05),
            "colsample_bytree" : trial.suggest_discrete_uniform('colsample_bytree', 0.65, 0.95, 0.05)
            }
        self.base_params_space.update(self.params_space)        
        self.base_params.update(self.params_space)
        model = gbdt_xgb.XGBReg(self.base_params)
        tr_x, va_x, tr_y, va_y = train_test_split(x, y, train_size = 0.9, shuffle = self.shuffle)
        model, score = model.train_and_evaluate(tr_x, va_x, tr_y, va_y)
        return - score
    
    def fetch_best_params(self, tr_x, tr_y):
        f = partial(self._objective, tr_x, tr_y)
        study = create_study()
        study.optimize(f, n_trials = 100)
        print('params:', study.best_params)
        self.base_params.update(study.best_params)
        return self.base_params

