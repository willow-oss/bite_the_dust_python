# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:25:32 2019

@author: willow-oss
"""

from learningalgos.algo_base import Algo_Base
from sklearn.linear_model import LinearRegression, Lasso

class Liniear(Algo_Base):
    def __init__(self, model_type = "Simple", alpha = 10e-7):
        if model_type == "Simple":
            self.model = LinearRegression()

        elif model_type == "Lasso":
            self.alpha = alpha
            self.model = Lasso(self.alpha)
    
    
    