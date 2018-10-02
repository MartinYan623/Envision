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

# Use grid-search to find the optimized parameters
def grid_search_XGB(train, predictors, fold=3):
    parameter_grid = {
        'max_depth': [2,3,4],
        'gamma':[0.1,0.2,0.3],
        'subsample':[0.6,0.7,0.8],
    }
    optimized_parameter = {
        'max_depth':0,
        'gamma': 0,
        'subsample':0
    }

    column_name = copy.deepcopy(predictors)
    column_name = column_name.append('Y.ws_tb')
    min_rmse = 999
    for a in range(len(parameter_grid['max_depth'])):
        for b in range(len(parameter_grid['gamma'])):
            for c in range(len(parameter_grid['subsample'])):
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

                    model = XGBRegressor(max_depth=parameter_grid['max_depth'][a], gamma=parameter_grid['gamma'][b],
                                         subsample=parameter_grid['subsample'][c])
                    model.fit(train_validation[predictors], train_validation['Y.ws_tb'])
                    sum_rmse += mean_squared_error(test_validation['Y.ws_tb'], model.predict(test_validation[predictors]))

                if min_rmse > sum_rmse/fold:
                    min_rmse = sum_rmse/fold
                    optimized_parameter['max_depth'] = parameter_grid['max_depth'][a]
                    optimized_parameter['gamma'] = parameter_grid['gamma'][b]
                    optimized_parameter['subsample'] = parameter_grid['subsample'][b]

    print('The best rmse score: ' + str(min_rmse))
    print(optimized_parameter)

