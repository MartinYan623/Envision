# coding=utf-8
import os
import sys
sys.path.append('../')
import logging
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from xgb_wswp.config import train_frequency, test_start_date, test_end_date, evaluate_frequency, \
    data_path, get_train_info
from power_forecast_common.common_misc import load_data_from_pkl, generate_folder
from mlApproach.util import get_nwp_list
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.wswp_error import write_wind_error, check_original_std, wswp_error_analysis
from power_forecast_common.evaluation_misc import get_training_data, calculate_rmse
from datetime import date

def calculate(feature_path, turbine_info, subsection=False):
    rmse = []
    for i in range(58):
        turbine_id = turbine_info.ix[i]['master_id']
        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))
        # print("calculating rmse for turbine {}".format(turbine_id))
        x_train, y_train = load_data_from_pkl(feature_file_path)
        if subsection == True:
            data = pd.concat([x_train, y_train['Y.ws_tb']], axis=1)
            data = data[(data['Y.ws_tb'] >= 3) & (data['Y.ws_tb'] <= 15)]
            rmse.append(calculate_rmse(np.array(data['Y.ws_tb']), np.array(data['prediction'])))
        else:
            rmse.append(calculate_rmse(np.array(y_train['Y.ws_tb']), np.array(x_train['prediction'])))
    return np.nanmean(np.array(rmse))

def calculate_baseline_rmse(x_df, y_df, revised_wd_df, subsection=False):

    data = pd.concat([x_df, y_df], axis=1)
    data = pd.concat([data, revised_wd_df], axis=1)
    data = data[(data['X_basic.horizon'] >= 16) & (data['X_basic.horizon'] <= 39)]

    if subsection == True:
        data = data[(data['Y.ws_tb'] >= 3) & (data['Y.ws_tb'] <= 15)]

    error_dict = {}
    nwp_list = get_nwp_list(data.columns.values)
    for nwp in nwp_list:
        org_error = calculate_rmse(np.array(data['Y.ws_tb']), np.array(data[nwp + ".ws"]))
        error_dict.update({nwp + "_org": org_error})

        cur_error = calculate_rmse(np.array(data['Y.ws_tb']), np.array(data[nwp + ".revised_ws"]))
        error_dict.update({nwp + "_revised": cur_error})

    return error_dict

if __name__ == '__main__':

    farm_id = "57f2a"

    # train_start_date, train_end_date = get_train_info(farm_id)
    # linear, ridge, lasso, elasticnet, svr, rf, xgb
    model = 'nn_new_sampling'
    model_type = 'model_revised_ws_shift_'+model+'_partial_training_resample'
    feature_type = "test_data_{}".format(model_type[6:])

    # add new code
    test_start_date = '2018-10-25'
    test_end_date = '2018-10-31'
    test_start_date = date(*map(int, test_start_date.split('-')))
    test_end_date = date(*map(int, test_end_date.split('-')))

    feature_path = generate_folder("result", feature_type, farm_id, test_start_date, test_end_date, train_frequency)
    # read farm_info file
    farm_info_path = '../data/farm_'+farm_id+'/farm_'+farm_id+'_info.csv'
    turbine_info = pd.read_csv(farm_info_path)
    print('the RMSE of 3-15m/s wind speed is: ' + str(calculate(feature_path, turbine_info, True)))
    print('the RMSE of all wind speed is: ' + str(calculate(feature_path, turbine_info)))
