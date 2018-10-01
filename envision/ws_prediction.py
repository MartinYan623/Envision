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
import copy
# load data
x_df, y_df=load_data_from_pkl('data/turbine_314e3ca4bd2345c1bc4f649f313d0b18.pkl')
#print(x_df)
#print(y_df)

# concat by column
data = pd.concat([x_df, y_df], axis=1)
#print(data)

EC0_predictors = ['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho']
GFS0_predictors = ['GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho']
WRF0_predictors = ['WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho']
predictors = ['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho', 'GFS0.ws', 'GFS0.wd', 'GFS0.tmp',
              'GFS0.pres', 'GFS0.rho', 'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho']


def EC0_prediction(data, start_day=1, end_day=394, probability=0.8, split=5, gride_seach=False, ensemble = False):
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

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: '+str(len(test['i.set'].unique())))
    if gride_seach:
        model = grid_search(train, EC0_predictors)

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
        model = grid_search(train, GFS0_predictors)

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
        model = grid_search(train, WRF0_predictors)
    else:
        if ensemble:
            ensemble_prediction(train, test, WRF0_predictors)
        else:
            single_prediction(train, test, WRF0_predictors)

# combine data of 3 whether stations (3x5=15)
def whole_prediction(data, start_day=1, end_day=394, probability=0.8, split=5, gride_seach=False, ensemble=False):
    data = data[np.isnan(data['GFS0.ws']) == False]
    data = data[np.isnan(data['WRF0.ws']) == False]
    train = pd.DataFrame(columns=['i.set', 'EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho',
                                  'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho',
                                  'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho',
                                  'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho',
                                  'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])

    for i in range(start_day, end_day, 1):
        data_1 = data[data['i.set'] == i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum() / 289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            a = random.uniform(0, 1)
            if a < 1 / split:
                test = pd.concat([test, data_1], names=['i.set', 'EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho',
                                  'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho',
                                  'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])
            else:
                train = pd.concat([train, data_1], names=['EC0.ws', 'EC0.wd', 'EC0.tmp', 'EC0.pres', 'EC0.rho',
                                  'GFS0.ws', 'GFS0.wd', 'GFS0.tmp', 'GFS0.pres', 'GFS0.rho',
                                  'WRF0.ws', 'WRF0.wd', 'WRF0.tmp', 'WRF0.pres', 'WRF0.rho', 'Y.ws_tb'])

    print('The number of training data set is: ' + str(len(train['i.set'].unique())))
    print('The number of testing data set is: ' + str(len(test['i.set'].unique())))

    if gride_seach:
        model = grid_search(train,predictors)
    else:
        if ensemble:
            ensemble_prediction(train, test, predictors)
        else:
            single_prediction(train, test, predictors)

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
    print(' RMSE of training set is: \n', mean_squared_error(y_train, clf.predict(x_train)))
    print (' RMSE of testing set is: \n', mean_squared_error(y_test, clf.predict(x_test)))

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
    print('RMSE of testing set is: \n', mean_squared_error(y_test, predictions))


# Use grid-search to find the optimized parameters
def grid_search(train, predictors, fold=3):
    parameter_grid = {
        'max_depth': [2,3,4],
        'gamma':[0.1,0.2,0.3]
    }
    optimized_parameter = {
        'max_depth':0,
        'gamma': 0
    }

    column_name = copy.deepcopy(predictors)
    column_name = column_name.append('Y.ws_tb')
    min_rmse = 999
    for a in range(len(parameter_grid['max_depth'])):
        for b in range(len(parameter_grid['gamma'])):
            unique_id = train['i.set'].unique()
            sum_rmse = 0
            for i in range(fold):
                train_validation = pd.DataFrame(columns=column_name)
                test_validation = pd.DataFrame(columns=column_name)
                for j in range(len(unique_id)):
                    if j % fold == i:
                        test_validation = pd.concat([test_validation, train[train['i.set'] == unique_id[j]]],names=[column_name])
                    else:
                        train_validation = pd.concat([train_validation, train[train['i.set'] == unique_id[j]]],
                                                     names=[column_name]
                                                     )

                model = XGBRegressor(max_depth=parameter_grid['max_depth'][a], gamma=parameter_grid['gamma'][b])
                model.fit(train_validation[predictors], train_validation['Y.ws_tb'])
                sum_rmse += mean_squared_error(test_validation['Y.ws_tb'], model.predict(test_validation[predictors]))

            if min_rmse > sum_rmse/fold:
                min_rmse = sum_rmse/fold
                print(min_rmse)
                optimized_parameter['max_depth'] = parameter_grid['max_depth'][a]
                optimized_parameter['gamma'] = parameter_grid['gamma'][b]

    print('The best score: ' + str(min_rmse))
    print(optimized_parameter)



GFS0_prediction(data, gride_seach=True)
"""
whole_prediction(data, ensemble= True)
EC0_prediction(data)
WRF0_prediction(data)
EC0_prediction(data, ensemble=True)
GFS0_prediction(data, ensemble=True)
WRF0_prediction(data, ensemble=True)
"""

