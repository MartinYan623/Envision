from common_misc import load_data_from_pkl
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
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import *


# load data
x_df, y_df=load_data_from_pkl('data/turbine_314e3ca4bd2345c1bc4f649f313d0b18.pkl')
#print(x_df)
#print(y_df)

# concat by column
data = pd.concat([x_df, y_df], axis=1)
#print(data)
#print(data['Y.ws_tb'].describe())
data=data.dropna(subset=['Y.ws_tb'])
#sns.distplot(data['Y.ws_tb'],fit=norm)
#plt.show()
#data['Y.ws_tb']=np.log(data['Y.ws_tb'])
#sns.distplot(data['Y.ws_tb'],fit=norm)
#plt.show()


EC0_predictors = ['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho']
GFS0_predictors = ['GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho']
WRF0_predictors = ['WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho']
predictors = ['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho', 'GFS0.ws', 'GFS0.wd', 'GFS0.tmp',
              'GFS0.pres', 'GFS0.rho', 'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho']


def EC0_prediction(data, start_day=1, end_day=394, probability=0.5, split=5, gride_seach=False, ensemble = False):
    train = pd.DataFrame(columns=['i.set','EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho', 'Y.ws_tb'])
    for i in range(start_day, end_day, 1):
        data_1 = data[data['i.set'] == i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum()/289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            # random split training data set and testing data set
            a = random.uniform(0,1)
            if a < 1/split:
                test = pd.concat([test, data_1], names=['i.set', 'EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres',
                                                      'EC0.rho', 'Y.ws_tb'])
            else:
                train = pd.concat([train, data_1], names=['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres',
                                                        'EC0.rho', 'Y.ws_tb'])
    train['Y.ws_tb'] = np.log(train['Y.ws_tb'])

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: '+str(len(test['i.set'].unique())))
    if gride_seach:
        model = grid_search_XGB(train, EC0_predictors)

    else:
        if ensemble:
            ensemble_prediction(train, test, EC0_predictors)
        else:
            single_prediction(train, test, EC0_predictors)

def GFS0_prediction(data, start_day=1, end_day=394, probability=0.8, split=5, gride_seach=False, ensemble = False):
    data = data[np.isnan(data['GFS0.ws']) == False]
    train = pd.DataFrame(columns=['i.set', 'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho', 'Y.ws_tb'])

    for i in range(start_day, end_day, 1):
        data_1=data[data['i.set'] == i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum()/289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            a = random.uniform(0, 1)
            if a < 1 / split:
                test = pd.concat([test, data_1], names=['i.set', 'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres',
                                                        'GFS0.rho', 'Y.ws_tb'])
            else:
                train = pd.concat([train, data_1], names=['GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres',
                                                          'GFS0.rho', 'Y.ws_tb'])
    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))
    if gride_seach:
        model = grid_search_XGB(train, GFS0_predictors)

    else:
        if ensemble:
            ensemble_prediction(train, test, GFS0_predictors)
        else:
            single_prediction(train, test, GFS0_predictors)

def WRF0_prediction(data, start_day=1, end_day=394, probability=0.8, split=5, gride_seach= False, ensemble = False):
    data = data[np.isnan(data['WRF0.ws']) == False]
    train = pd.DataFrame(columns=['i.set', 'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])
    for i in range(start_day, end_day, 1):
        data_1 = data[data['i.set'] == i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum() / 289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            a = random.uniform(0, 1)
            if a < 1 / split:
                test = pd.concat([test, data_1], names=['i.set', 'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres',
                                                        'WRF0.rho', 'Y.ws_tb'])
            else:
                train = pd.concat([train, data_1], names=['WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres',
                                                          'WRF0.rho', 'Y.ws_tb'])
    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))

    if gride_seach:
        model = grid_search_XGB(train, WRF0_predictors)
    else:
        if ensemble:
            ensemble_prediction(train, test, WRF0_predictors)
        else:
            single_prediction(train, test, WRF0_predictors)



# regression prediction single model
def single_prediction(train, test, predictors):
    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    # GradientBoost regression
    #clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, subsample=0.8, min_samples_split=2,
    #                                min_samples_leaf=3, max_depth=2, alpha=0.9)
    # simple linear regression
    # clf = linear_model.LinearRegression()

    # XGBoost regression
    clf = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, gamma=0.1, subsample=0.8)
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

    clf.fit(x_train, y_train)
    print(' RMSE of training set is: \n', np.sqrt(mean_squared_error(np.exp(y_train),  np.exp(clf.predict(x_train)))))
    print (' RMSE of testing set is: \n', np.sqrt(mean_squared_error(y_test, np.exp(clf.predict(x_test)))))

# regression prediction ensemble model
def ensemble_prediction(train, test, predictors):

    x_train = train[predictors]
    y_train = train['Y.ws_tb']
    x_test = test[predictors]
    y_test = test['Y.ws_tb']

    algorithms = [
    XGBRegressor(n_estimators=200, max_depth=2, reg_lambda=0.8, learning_rate=0.1, gamma=0.1, subsample=0.8, eta=0.01,
                       early_stopping_rounds=100, colsample_bytree=0.8),
    RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=6, max_depth=2, random_state=1,
                              n_jobs= -1),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=200, subsample=0.8, min_samples_split=2,
                                    min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_depth=2, alpha=0.9),

    linear_model.LinearRegression(),
    linear_model.Ridge(alpha=0.01),
    linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100)),
    linear_model.Lasso(alpha=0.01),
    linear_model.ElasticNet(l1_ratio=0.2)
    ]

    full_predictions = []
    for clf in algorithms:
        model = clf.fit(x_train, y_train)
        predictions = model.predict(x_test)
        full_predictions.append(predictions)
    predictions = (full_predictions[0]*0.6 + full_predictions[1]*0.2 + full_predictions[2]*0.2)
    print('RMSE of testing set is: \n', np.sqrt(mean_squared_error(y_test, predictions)))


EC0_prediction(data)
"""
GFS0_prediction(data, gride_seach=True)
WRF0_prediction(data)
EC0_prediction(data, ensemble=True)
GFS0_prediction(data, ensemble=True)
WRF0_prediction(data, ensemble=True)
"""

