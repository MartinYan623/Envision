from common_misc import load_data_from_pkl
from evaluation_misc import wind_std,wind_std_distribution,calculate_mbe
from preprocess import preprocess
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import random
from grid_search import grid_search_XGB
from sklearn.linear_model import BayesianRidge
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import *
from numpy.random import seed
seed(0)

"""
'EC0.tmp_0.0', 'EC0.tmp_1.0', 'EC0.tmp_2.0', 'EC0.tmp_3.0', 'EC0.tmp_4.0', 'EC0.tmp_5.0',

'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_12',
  'GFS0.wd_0.0', 'GFS0.wd_1.0', 'GFS0.wd_2.0', 'GFS0.wd_3.0', 'GFS0.wd_4.0', 'GFS0.wd_5.0',
    'season_spring', 'season_summer', 'season_winter',
"""

predictors = ['EC0.ws', 'EC0.wd','EC0.tmp','EC0.pres', 'EC0.rho', 'GFS0.ws','GFS0.wd',
              'GFS0.tmp', 'GFS0.pres', 'GFS0.rho', 'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho',
              'time_morning', 'time_afternoon', 'time_night']

predictors = ['EC0.ws','EC0.wd', 'EC0.pres', 'EC0.tmp','EC0.rho',
              'GFS0.ws','GFS0.wd','GFS0.tmp', 'GFS0.pres', 'GFS0.rho',
              'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho' ]

# combine data of 3 whether stations (3x5=15)
def whole_prediction(train, test, ensemble=False):
    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))

    if ensemble:
        std_train, std_test = ensemble_prediction(train, test, predictors)
    else:
        std_train, std_test = single_prediction(train, test, predictors)

    return std_train,std_test

# regression prediction single model
def single_prediction(train, test, predictors):
    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    # GradientBoost regression
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=0.8, max_depth=2, alpha=0.9)
    # simple linear regression
    #clf = linear_model.LinearRegression()
    # XGBoost regression
    # clf = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.1, gamma=0.1, subsample=0.6)
    # Random forest regression
    #clf = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=6, max_depth=3, random_state=1, n_jobs=-1)
    # Ridge regression
    # clf = linear_model.Ridge(alpha=0.01)
    # Ridge regression
    # clf = linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100))
    # Lasso regression
    #clf = linear_model.Lasso(alpha=0.01)
    # Flexible network (combined Lasso regression and Ridge regression)
    # clf = linear_model.ElasticNet(l1_ratio=0.2)
    # clf = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
    #clf = SVR()
    # clf = BayesianRidge(n_iter=200, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
    clf.fit(x_train, y_train)
    print(' std of training set is: \n', round(wind_std(y_train, clf.predict(x_train), mean_bias_error=None), 5))
    print(' std of testing set is: \n', round(wind_std(y_test,clf.predict(x_test), mean_bias_error=None), 5))
    std_train=round(wind_std(y_train, clf.predict(x_train), mean_bias_error=None), 5)
    std_test=round(wind_std(y_test,clf.predict(x_test), mean_bias_error=None), 5)

    return std_train,std_test

def ensemble_prediction(train, test, predictors):

    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    algorithms = [
    XGBRegressor(n_estimators=200, max_depth=2, reg_lambda=0.8, learning_rate=0.1, gamma=0.1, subsample=0.8, eta=0.01,
                       early_stopping_rounds=100, colsample_bytree=0.8),
    linear_model.LinearRegression(),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, subsample=0.8, max_depth=2, alpha=0.9),

    linear_model.Ridge(alpha=0.01),
    linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100)),
    linear_model.Lasso(alpha=0.01),
    linear_model.ElasticNet(l1_ratio=0.2)
    ]

    full_predictions1 = []
    full_predictions2 = []
    for clf in algorithms:
        model = clf.fit(x_train, y_train)
        predictions1 = model.predict(x_train)
        predictions2 = model.predict(x_test)
        full_predictions1.append(predictions1)
        full_predictions2.append(predictions2)
    predictions1 = (full_predictions1[0] * 0.4 + full_predictions1[1] * 0.3 + full_predictions1[2] * 0.3)
    predictions2 = (full_predictions2[0] * 0.4 + full_predictions2[1] * 0.3 + full_predictions2[2] * 0.3)
    print(' std of training set is: \n', wind_std(y_train, predictions1, mean_bias_error=None))
    print(' std of testing set is: \n', wind_std(y_test, predictions2, mean_bias_error=None))
    std_train = wind_std(y_train,predictions1, mean_bias_error=None)
    std_test = wind_std(y_test,  predictions2, mean_bias_error=None)

    return std_train,  std_test

sum_std_train=0
sum_std_test=0
for i in range(10):

    print('load data set '+str(i+1))

    # load data
    x_train, y_train=load_data_from_pkl('data/turbine_%s_train.pkl' % str(i+1))
    # test data include one month data
    x_test, y_test=load_data_from_pkl('data/turbine_%s_test.pkl' % str(i+1))

    print('#' * 33)
    print('start data pre-processing')
    print('#' * 33)
    # pre-process data
    data_train, data_test = preprocess(x_train, y_train, x_test, y_test)

    print('#' * 33)
    print('start prediction')
    print('#' * 33)
    # build model and prediction
    std_train, std_test = whole_prediction(data_train, data_test)
    sum_std_train += std_train
    sum_std_test += std_test

print('mean std of training data: '+str(sum_std_train/10))
print('mean std of testing data: '+str(sum_std_test/10))