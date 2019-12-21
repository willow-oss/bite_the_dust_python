# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:50:26 2019

@author: willow-oss
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer    

class Algo_Base():
    def __init__(self):
        pass
    
    def transform(self, x):
        tr = QuantileTransformer(n_quantiles = len(x), 
                                      random_state = 0, 
                                      output_distribution = "normal")
        x = pd.DataFrame(data = self.tr.fit_transform(x), columns = x.columns)
        return tr, x

    def train(self, tr_x, tr_y):
        self.model.fit(tr_x, tr_y)
    
        
    def predict(self, va_x):
        return self.model.predict(va_x)        
    
    def train_and_evaluate(self, tr_x, va_x, tr_y, va_y, 
                           plot_learning_curve = False, 
                           plot_validation_scatter = False):
        self.train( tr_x, tr_y)
        pred_y = self.predict(va_x)
        score = r2_score(va_y, pred_y)
        print("R-squared on validation data is " + '{:.2g}'.format(score))
        if plot_validation_scatter :
            self.plot_result(va_y, pred_y)
        return self.model, score        
    
    def plot_result(self, va_y, pred_y):
        sns.set()
        plot = sns.jointplot(x = va_y, y = pred_y, kind="reg", stat_func= r2_score)
        plot.set_axis_labels('va_y', 'pred_y')
        plt.title(self.name + " model regression", y=1.2, fontsize = 16)
        