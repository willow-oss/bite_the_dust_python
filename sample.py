# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:46:09 2019

@author: Haru
"""

from LoadTestData import TestData
from LearningAlgos import ExtraTrees, GBDT_XGB
from HypOptParams import ETOpt, XGBOpt
from sklearn.model_selection import train_test_split

b = TestData.boston_data_df()
df = b.fetch_boston_df()

X = df.drop("PRICE", axis = 1)
y = df["PRICE"]

tr_X, va_X, tr_y, va_y = train_test_split(X, y, train_size = 0.9)


"""
#Extra Trees test mode
ETR = ExtraTrees.ETReg()
base_params = ETR.base_params

ETO = ETOpt.ETRegOpt()
best_params = ETO.fetch_best_params(tr_X, va_X, tr_y, va_y)
print(best_params)
"""

"""
#xgboost test mode
XGBR = GBDT_XGB.XGBReg()
base_params = XGBR.base_params

XGBO = XGBOpt.XGBRegOpt()
best_params = XGBO.fetch_best_params(tr_X, va_X, tr_y, va_y)
XGBR = GBDT_XGB.XGBReg(best_params)
XGBR.train_and_predict(tr_X, va_X, tr_y, va_y, plot_bool = True)
"""