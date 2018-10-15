from common_misc import load_data_from_pkl
from evaluation_misc import wind_std,wind_std_distribution,calculate_mbe
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


EC0_predictors = ['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho']
GFS0_predictors = ['GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho']
WRF0_predictors = ['WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho']
predictors = ['EC0_prediction','GFS0_prediction','WRF0_prediction']

def EC0_prediction(train, test, gride_seach=False, ensemble = False):

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: '+str(len(test['i.set'].unique())))
    if gride_seach:
        parameters = grid_search_XGB(train, EC0_predictors)
        std_train, std_test = single_prediction(train, test, EC0_predictors, parameters)
    else:
        if ensemble:
            std_train, std_test = ensemble_prediction(train, test, EC0_predictors)
        else:
            std_train, std_test = single_prediction(train, test, EC0_predictors)

    return std_train, std_test


def GFS0_prediction(train, test, gride_seach=False, ensemble=False):

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))
    if gride_seach:
        parameters = grid_search_XGB(train, GFS0_predictors)
        std_train, std_test = single_prediction(train, test, GFS0_predictors,parameters)
    else:
        if ensemble:
            std_train, std_test =  ensemble_prediction(train, test, GFS0_predictors)
        else:
            std_train, std_test = single_prediction(train, test, GFS0_predictors)
    return std_train, std_test

def WRF0_prediction(train, test, gride_seach=False, ensemble = False):

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))

    if gride_seach:
        parameters = grid_search_XGB(train, WRF0_predictors)
        std_train, std_test = single_prediction(train, test, WRF0_predictors, parameters)
    else:
        if ensemble:
            std_train, std_test= ensemble_prediction(train, test, WRF0_predictors)
        else:
            std_train, std_test= single_prediction(train, test, WRF0_predictors)

    return std_train, std_test

def single_prediction(train, test, predictors, parameters=None):
    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    # GradientBoost regression
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=0.8, max_depth=2,
                                    alpha=0.9)
    # simple linear regression
    # clf = linear_model.LinearRegression()
    # XGBoost regression
    # clf = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, gamma=0.1, subsample=0.6)
    # Random forest regression
    # clf = RandomForestRegressor(n_estimators=300, criterion='mse', min_samples_leaf=6, max_depth=5, random_state=1, n_jobs=-1)
    # Ridge regression
    # clf = linear_model.Ridge(alpha=0.01)
    # Ridge regression
    # clf = linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100))
    # Lasso regression
    # clf = linear_model.Lasso(alpha=0.01)
    # Flexible network (combined Lasso regression and Ridge regression)
    # clf = linear_model.ElasticNet(l1_ratio=0.2)
    # clf = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
    # clf = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
    # clf = BayesianRidge(n_iter=200, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
    clf.fit(x_train, y_train)
    #print(' std of training set is: \n', round(wind_std(y_train, clf.predict(x_train), mean_bias_error=None),5))
    #print(' std of testing set is: \n', round(wind_std(y_test, clf.predict(x_test), mean_bias_error=None),5))

    prediction_train = clf.predict(x_train)
    prediction_test = clf.predict(x_test)

    return prediction_train, prediction_test


def ensemble_prediction(train, test, predictors):

    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    algorithms = [
    XGBRegressor(n_estimators=200, max_depth=2, reg_lambda=0.8, learning_rate=0.1, gamma=0.1, subsample=0.8),
    RandomForestRegressor(n_estimators=200, criterion='mse',max_depth=2, random_state=1, n_jobs= -1),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, subsample=0.8, max_depth=2, alpha=0.9),

    linear_model.LinearRegression(),
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

    print(' std of testing set is: \n', wind_std(y_train, predictions1, mean_bias_error=None))
    print(' std of testing set is: \n', wind_std(y_test, predictions2, mean_bias_error=None))

    std_train = wind_std(y_train, predictions1, mean_bias_error=None)
    std_test = wind_std(y_test, predictions2, mean_bias_error=None)

    return std_train,std_test


def combining_model(train,test):
    print('#' * 33)
    print('start combining model')
    print('#' * 33)

    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    print(' std of training set is: \n', round(wind_std(y_train, clf.predict(x_train), mean_bias_error=None),5))
    print(' std of testing set is: \n', round(wind_std(y_test,clf.predict(x_test), mean_bias_error=None),5))

def smooth_Y(data):
    smooth_y=[]
    for i in range(394):
        sub_data=data[data['i.set']==i]
        a=np.array(sub_data['Y.ws_tb'].reshape(289,1))
        new_a= [a[0][0],a[1][0],a[2][0]]
        for j in range(3,len(a)-2):
            if np.isnan(a[j])==False:
                # 'nanmean' skips nan value
                new_a.append(np.nanmean(a[j-3:j+3,:]))
            else:
                new_a.append(np.nan)
        new_a.append(a[287][0])
        new_a.append(a[288][0])
        smooth_y=smooth_y+new_a
    return smooth_y

sum_EC0_std_train=0
sum_EC0_std_test=0
sum_GFS0_std_train=0
sum_GFS0_std_test=0
sum_WRF0_std_train=0
sum_WRF0_std_test=0

for i in range(10):

    print('load data set '+str(i+1))
    # load data
    x_train, y_train=load_data_from_pkl('data/turbine_%s_train.pkl' % str(i+1))
    # test data include one month data
    x_test, y_test=load_data_from_pkl('data/turbine_%s_test.pkl' % str(i+1))

    # concat by column
    data_train = pd.concat([x_train, y_train], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)

    # whether smooth data_y
    # data_train['Y.ws_tb']=smooth_Y(data_train)

    data_train = data_train.dropna(subset=['Y.ws_tb'])
    data_train = data_train[np.isnan(data_train['GFS0.ws']) == False]
    data_train = data_train[np.isnan(data_train['WRF0.ws']) == False]
    data_test = data_test.dropna(subset=['Y.ws_tb'])

    EC0_prediction_train, EC0_prediction_test = EC0_prediction(data_train, data_test)
    GFS0_prediction_train, GFS0_prediction_test = GFS0_prediction(data_train, data_test)
    WRF0_prediction_train, WRF0_prediction_test = WRF0_prediction(data_train, data_test)


    data_train = pd.DataFrame({'EC0_prediction':EC0_prediction_train, 'GFS0_prediction':GFS0_prediction_train,
                               'WRF0_prediction':WRF0_prediction_train, 'Y.ws_tb': data_train['Y.ws_tb']})
    data_test = pd.DataFrame({'EC0_prediction': EC0_prediction_test, 'GFS0_prediction': GFS0_prediction_test,
                               'WRF0_prediction': WRF0_prediction_test,'Y.ws_tb': data_test['Y.ws_tb']})
    combining_model(data_train, data_test)

