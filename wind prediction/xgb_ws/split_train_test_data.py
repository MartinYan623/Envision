# coding=utf-8
import os
import sys
sys.path.append('../')
import logging
from sklearn.externals import joblib
from datetime import datetime, date
import pandas as pd
import numpy as np
from xgb_wswp.config import data_path, get_train_info, train_frequency
from power_forecast_common.common_misc import load_data_from_pkl, generate_folder
from mlApproach.util import get_nwp_list
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.wswp_error import write_wind_error, check_original_std, wswp_error_analysis
from power_forecast_common.evaluation_misc import get_training_data
from power_forecast_common.offline_common import filter_data

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # generate training and testing data
    farm_id = "57f2a"
    train_start_date = '2017-10-04'
    train_end_date = '2018-10-17'
    test_start_date = '2018-10-18'
    test_end_date = '2018-10-24'
    train_start_date = date(*map(int, train_start_date.split('-')))
    train_end_date = date(*map(int, train_end_date.split('-')))
    test_start_date = date(*map(int, test_start_date.split('-')))
    test_end_date = date(*map(int, test_end_date.split('-')))

    train_end = '2018-10-18 00:00:00'
    # test data time one week
    test_start = '2018-10-20 00:00:00'
    test_end = '2018-10-25 00:00:00'

    train_data_path = generate_folder(data_path, "train_data_IBM_5", farm_id, train_start_date, train_end_date, '60min')
    test_data_path = generate_folder(data_path, "test_data_IBM_5", farm_id, test_start_date, test_end_date, '60min')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    input_data_path = generate_folder(data_path, "data_all_nwp", farm_id,  date(*map(int, '2017-10-04'.split('-'))), \
                                      date(*map(int, '2018-11-08'.split('-'))), '60min')
    # read farm_info file
    farm_info_path = '../data/farm_' + farm_id + '/farm_' + farm_id + '_info.csv'
    turbine_info = pd.read_csv(farm_info_path)

    for i in range(58):
        turbine_id = turbine_info.ix[i]['master_id']
        turbine_file_path = os.path.join(input_data_path, "turbine_{}.pkl".format(turbine_id))
        train_file_path = os.path.join(train_data_path, "turbine_{}.pkl".format(turbine_id))
        test_file_path = os.path.join(test_data_path, "turbine_{}.pkl".format(turbine_id))

        x_df, y_df = load_data_from_pkl(turbine_file_path)
        x_train, y_train = filter_data(x_df, y_df, train_end)
        x_test, y_test = filter_data(x_df, y_df, test_end, test_start)
        feature_table = pd.concat([x_train, y_train], axis=1)
        feature_table.to_pickle(train_file_path)
        feature_table = pd.concat([x_test, y_test], axis=1)
        feature_table.to_pickle(test_file_path)