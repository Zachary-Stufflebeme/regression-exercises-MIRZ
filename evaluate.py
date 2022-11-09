from math import sqrt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from grab_db import my_db
import os
import matplotlib.pyplot as plt

def plot_residuals(y,yhat):
    plt.scatter(y,yhat)
    plt.show()

def regression_errors(y,yhat):
    residual = yhat - y
    residual_sq = residual ** 2
    SSE = sum(residual_sq)
    MSE = SSE/len(y)
    RMSE = sqrt(MSE)
    ESS = sum((residual - y.mean())**2)
    TSS = ESS + SSE
    return('SSE = ',SSE,'MSE = ',MSE, 'RMSE = ',RMSE,'ESS = ',ESS,'TSS = ',TSS)
def baseline_mean_errors(y):
    baseline = y.mean()
    residual = baseline - y
    residual_sq = residual ** 2
    SSE = sum(residual_sq)
    MSE = SSE/len(y)
    RMSE = sqrt(MSE)
    return('SSE = ',SSE,'MSE = ',MSE,'RMSE = ',RMSE)


def better_than_baseline(y,yhat):
    a,b,c,d,e,f,g,h,i,j = regression_errors(y,yhat)
    k,l,m,n,o,p = baseline_mean_errors(y)
    if f < p:
        return( True )
    else:
        return( False )






