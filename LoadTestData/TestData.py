# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:41:24 2019

@author: Haru
"""

from sklearn.datasets import load_boston
import pandas as pd

class boston_data_df():
    def __init__(self):
        pass
    
    def fetch_boston_df(self):
        boston = load_boston()
        boston_df = pd.DataFrame(boston.data)
        boston_df.columns = boston.feature_names
        boston_df['PRICE'] = pd.DataFrame(boston.target)
        
        return boston_df
