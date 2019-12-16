# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:46:09 2019

@author: Haru
"""

from LoadTestData import TestData
from LearningAlgos import ExtraTrees
from HypOptParams import ETOpt
from sklearn.model_selection import train_test_split

b = TestData.boston_data_df()
df = b.fetch_boston_df()

X = df.drop("PRICE", axis = 1)
y = df["PRICE"]

tr_X, va_X, tr_y, va_y = train_test_split(X, y, train_size = 0.9)

ETR = ExtraTrees.ETReg()
base_params = ETR.base_params

ETO = ETOpt.ETRegOpt()
best_params = ETO.fetch_best_params(tr_X, va_X, tr_y, va_y)
print(best_params)