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

def calculate(feature_path, turbine_info):
    rmse = []
    for i in range(58):
        print(i)
        turbine_id = turbine_info.ix[i]['master_id']
        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))
        print("calculating rmse for turbine {}".format(turbine_id))
        x_train, y_train = load_data_from_pkl(feature_file_path)
        rmse.append(calculate_rmse(np.array(y_train['Y.ws_tb']), np.array(x_train['prediction'])))
    print(np.mean(np.array(rmse)))

if __name__ == '__main__':

    farm_id = "57f2a7f2a624402c9565e51ba8d171cb"

    train_start_date, train_end_date = get_train_info(farm_id)

    # baseline, linear, ridge, lasso, elasticnet, svr, rf, xgb
    model = 'xgb2'
    model_type = 'model_revised_ws_shift_'+model+'_partial_training_resample'
    feature_type = "test_data_{}".format(model_type[6:])

    feature_path = generate_folder("result", feature_type, farm_id, test_start_date, test_end_date, train_frequency)

    # read farm_info file
    farm_info_path = '../data/farm_'+farm_id+'/farm_'+farm_id+'_info.csv'
    turbine_info = pd.read_csv(farm_info_path)

    calculate(feature_path, turbine_info)