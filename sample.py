# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:46:09 2019

@author: willow-oss
"""

from testdataloader import boston_df
from learningalgos import extra_trees, gbdt_xgb, neural_net, gaussian_process
from hypparamsopt import et_opt, xgb_opt
from sklearn.model_selection import train_test_split

def main():

    b = boston_df.Boston_df()
    df = b.fetch_boston_df()
    
    x = df.drop("PRICE", axis = 1)
    y = df["PRICE"]
    
    tr_x, va_x, tr_y, va_y = train_test_split(x, y, train_size = 0.9)
    """
    #Extra Trees test mode
    etr = extra_trees.ETReg()
    base_params = etr.base_params
    
    eto = et_opt.ETRegOpt()
    best_params = eto.fetch_best_params(tr_x, tr_y)
    base_params.update(best_params)
    
    etr_best = extra_trees.ETReg(base_params)
    etr_best.train_and_predict(tr_x, va_x, tr_y, va_y)
    
    
    #xgboost test mode
    xgbr = gbdt_xgb.XGBReg()
    base_params = xgbr.base_params
    
    xgbo = xgb_opt.XGBRegOpt()
    best_params = xgbo.fetch_best_params(tr_x, tr_y)
    base_params.update(best_params)
    xgbr_best = gbdt_xgb.XGBReg(base_params)
    xgbr_best.train_and_predict(tr_x, va_x, tr_y, va_y, plot_learning_curve = True)
    """
    """
    #MLP test mode
    mlpr = neural_net.MLPReg()
    mlpr.train_and_evaluate(tr_x, va_x, tr_y, va_y)
    """
    
    
    #gaussian process test mode
    gpyr = gaussian_process.GPReg()
    gpyr.train_and_evaluate(tr_x, va_x, tr_y, va_y) 
        


if __name__ == "__main__":
    main()