import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 
import scipy.stats as stats
import math
import sklearn.preprocessing
from env import get_db_url

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def make_location_model_df(df,theme, n_clust):
    Scaler = MinMaxScaler()
    df[theme] = Scaler.fit_transform(df[theme])
    new = df[theme]
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(new)
    new['cluster'] = kmeans.predict(new)
    new[['LCluster0','LCluster1','LCluster2','LCluster3','LCluster4']] = pd.get_dummies(new['cluster'])
    new = new.drop(columns = ['LCluster0','LCluster3','LCluster4','cluster'])
    return(new)

def make_foundation_model_df(df,theme, n_clust):
    Scaler = MinMaxScaler()
    df[theme] = Scaler.fit_transform(df[theme])
    new = df[theme]
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(new)
    new['cluster'] = kmeans.predict(new)
    new[['FCluster0','FCluster1','FCluster2','FCluster3']] = pd.get_dummies(new['cluster'])
    new = new.drop(columns = ['FCluster0','FCluster1','FCluster3','cluster'])
    return(new)

#def regression_functions(
    
    
    
    
    
def baseline(df):  
    logerror_pred_median = df['logerror'].median()
    df['logerror_pred_median'] = logerror_pred_median
# RMSE of prop_value_pred_median
    rmse_baseline_train = mean_squared_error(df.logerror, df.logerror_pred_median)**(1/2)
#printing results of baseline model
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_baseline_train, 2),)