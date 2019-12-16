# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:07:41 2019

@author: Haru
"""

from optuna import trial, create_study
from functools import partial
from LearningAlgos import ExtraTrees

class ETRegOpt():
    
    def __init__(self):
        self.base_params = {"n_estimators":1000,
                                   "criterion":'mae',
                                   "max_features":'auto',
                                   "max_depth":None,
                                   "bootstrap":True,
                                   "min_samples_split":4,
                                   "min_samples_leaf":1,
                                   "min_weight_fraction_leaf":0,
                                   "max_leaf_nodes":None}
        
    def objective(self, tr_X, va_X, tr_y, va_y, trial):
        self.params_space = {"max_features" : trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
                             "max_depth" : trial.suggest_int("max_depth", 3, 9),
                             "bootstrap" : trial.suggest_categorical("bootstrap", [True, False]),
                             "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                             "min_weight_fraction_leaf": trial.suggest_uniform("min_weight_fraction_leaf", 0, 0.3)
                             }
        
        self.base_params.update(self.params_space)
        model = ExtraTrees.ETReg(self.base_params)
        model, score = model.train_and_predict(tr_X, va_X, tr_y, va_y)
        return - score
    
    def fetch_best_params(self, tr_X, va_X, tr_y, va_y):
        f = partial(self.objective, tr_X, va_X, tr_y, va_y)
        study = create_study()
        study.optimize(f, n_trials = 100)
        print('params:', study.best_params)
        self.base_params.update(study.best_params)
        return self.base_params
        
        
    
        