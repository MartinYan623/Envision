import pandas as pd
from evaluation_misc import wind_std,wind_std_distribution,calculate_mbe
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

# Use grid-search to find the optimized parameters
def grid_search_XGB(train, predictors, fold=6):
    parameter_grid = {
        'max_depth': [2,3,4],
        'n_estimators':[100,150,200,250],

    }
    optimized_parameter = {
        'max_depth':0,
        'n_estimators': 0,

    }

    column_name = copy.deepcopy(predictors)
    column_name = column_name.append('Y.ws_tb')
    min_std = 999
    for a in range(len(parameter_grid['max_depth'])):
        for b in range(len(parameter_grid['n_estimators'])):
            print('start to validate parameter max_depth: '+str(parameter_grid['max_depth'][a])+
                  ' and parameter n_estimators: '+str(parameter_grid['n_estimators'][b]))
            unique_id = train['i.set'].unique()
            sum_std = 0
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
                model = XGBRegressor(max_depth=parameter_grid['max_depth'][a], n_estimators=parameter_grid['n_estimators'][b],
                                     loss='ls', learning_rate=0.1, subsample=0.8, alpha=0.9)
                model.fit(train_validation[predictors], train_validation['Y.ws_tb'])
                sum_std += round(wind_std(test_validation['Y.ws_tb'], model.predict(test_validation[predictors]), mean_bias_error=None),5)
            print('std is: '+str(sum_std/fold))
            if min_std > sum_std/fold:
                min_std = sum_std/fold
                optimized_parameter['max_depth'] = parameter_grid['max_depth'][a]
                optimized_parameter['n_estimators'] = parameter_grid['n_estimators'][b]

    print('The best std score: ' + str(min_std))
    print(optimized_parameter)
    return optimized_parameter

