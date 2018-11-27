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
from xgb_ws.xgb_ws_forecast import XgbWsForecast
from xgb_ws.xgb_linear_ws_forecast import XgbLinearWsForecast
from xgb_ws.xgb_ridge_ws_forecast import XgbRidgeWsForecast
from xgb_ws.xgb_lasso_ws_forecast import XgbLassoWsForecast
from xgb_ws.xgb_elasticnet_ws_forecast import XgbElasticNetWsForecast
from xgb_ws.xgb_nn_ws_forecast import XgbNNWsForecast

logger = logging.getLogger(__name__)


def train_turbine_ws_model(master_id, lat, lon, turbine_data_path, feature_file_path, data_resampling=False,
                           train_frequency=60, delta_hour=3):
    """
    :param master_id:
    :param lat:
    :param lon:
    :param turbine_data_path:
    :param feature_file_path:
    :param train_frequency:
    :param delta_hour:
    :return:
    """
    logger.info('------Training model for wtg {}------'.format(master_id))

    #model = XgbWsForecast(master_id, lat=lat, lon=lon, grid_params=None)
    #model = XgbLinearWsForecast(master_id, lat=lat, lon=lon, grid_params=None)
    #model = XgbRidgeWsForecast(master_id, lat=lat, lon=lon, grid_params=None)
    #model = XgbLassoWsForecast(master_id, lat=lat, lon=lon, grid_params=None)
    #model = XgbElasticNetWsForecast(master_id, lat=lat, lon=lon, grid_params=None)
    model = XgbNNWsForecast(master_id, lat=lat, lon=lon, grid_params=None)

    assert turbine_data_path[-3:] == "pkl", "Unknown data file type!"
    x_df, y_df = load_data_from_pkl(turbine_data_path)
    if "Y.power_tb_revised" in y_df.columns:
        y_df["Y.power_tb"] = y_df["Y.power_tb_revised"]
        y_df.drop("Y.power_tb_revised", axis=1, inplace=True)

    if np.sum(np.isnan(y_df["Y.power_tb"])) >= 0.9 * len(y_df):
        return None

    nwp_list = get_nwp_list(x_df.columns.values)
    model.set_nwp_list(nwp_list)
    # can change training type to change the number of training data
    x_df, y_df = get_training_data(x_df, y_df, 1)

    # feature engineering
    feature_ins = WsWpFeature(train_frequency, delta_hour, nwp_list)
    x_df, feature_dict = feature_ins.transform(x_df)

    grid_params = {'silent': [1], 'eta': [0.05], 'max_depth': range(3, 5), 'min_child_weight': [1, 3],
                   'subsample': [0.5], 'lambda': [1]}
    model.configuration(train_frequency=train_frequency, grid_params=grid_params, data_resampling=data_resampling,
                        max_trees=500)
    x_df, nn_model = model.fit(x_df, y_df, feature_dict)
    feature_table = pd.concat([x_df, y_df], axis=1)
    feature_table.to_pickle(feature_file_path)

    wind_std_dict = check_original_std(x_df, y_df, nwp_list)
    wswp_error_analysis(model, y_df)
    model.update_error(wind_std_dict)
    return model, nn_model

"""
def train_farm(farm_id, train_data_path, model_path, feature_path, data_resampling=False, turbine_index=[]):

    farm_ins = Farm(farm_id)
    wtg_list = farm_ins.wtg_list
    if len(turbine_index) == 0:
        turbine_index = list(range(len(wtg_list)))
    for n, wtg_ins in enumerate(wtg_list):
        turbine_id = wtg_ins.master_id
        if n not in turbine_index:
            continue
        lon = wtg_ins.lon
        lat = wtg_ins.lat
        turbine_file_path = os.path.join(train_data_path, "turbine_{}.pkl".format(turbine_id))
        model_file_path = os.path.join(model_path, "turbine_{}.bin".format(turbine_id))
        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))
        if not os.path.exists(turbine_file_path) or os.path.exists(model_file_path):
            continue

        print("training for turbine {}".format(turbine_id))
        model = train_turbine_ws_model(turbine_id, lon=lon, lat=lat, feature_file_path=feature_file_path,
                                       turbine_data_path=turbine_file_path, data_resampling=data_resampling)
        if model is None:
            print("No trained model for turbine {}".format(turbine_id))
            continue
        joblib.dump(model, model_file_path)
        wind_error_file = os.path.join(model_path, "turbine_{}_train_wind_error.csv".format(turbine_id))
        write_wind_error(model.get_train_error(), wind_error_file)
"""

def train_farm_local(train_data_path, model_path, feature_path,  turbine_info, data_resampling=False):

    for i in range(58):
        print(i)
        turbine_id = turbine_info.ix[i]['master_id']
        lat = turbine_info.ix[i]['lat']
        lon = turbine_info.ix[i]['lon']

        turbine_file_path = os.path.join(train_data_path, "turbine_{}.pkl".format(turbine_id))
        model_file_path = os.path.join(model_path, "turbine_{}.bin".format(turbine_id))
        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))

        if not os.path.exists(turbine_file_path) or os.path.exists(model_file_path):
            print('No File')

        print("training for turbine {}".format(turbine_id))

        # add new code
        model, nn_model = train_turbine_ws_model(turbine_id, lon=lon, lat=lat, feature_file_path=feature_file_path,
                                   turbine_data_path=turbine_file_path, data_resampling=data_resampling)
        if model is None:
            print("No trained model for turbine {}".format(turbine_id))

        joblib.dump(model, model_file_path)
        # add new code
        nn_model.save(model_file_path[:-4] + '.h5')

        wind_error_file = os.path.join(model_path, "turbine_{}_train_wind_error.csv".format(turbine_id))
        write_wind_error(model.get_train_error(), wind_error_file)

if __name__ == '__main__':

    farm_id = "57f2a"
    # train_start_date, train_end_date = get_train_info(farm_id)
    # for appointed training set
    train_start_date = '2017-10-04'
    train_end_date = '2018-10-24'
    train_start_date = date(*map(int, train_start_date.split('-')))
    train_end_date = date(*map(int, train_end_date.split('-')))

    data_resampling = True

    # baseline, linear, ridge, lasso, elasticnet, nn
    model = 'nn_new_sampling'
    model_type = 'model_revised_ws_shift_'+model+'_partial_training_resample'
    feature_type = "train_data_{}".format(model_type[6:])

    train_data_path = generate_folder(data_path, "train_data_IBM_5", farm_id, train_start_date, train_end_date, train_frequency)
    model_path = generate_folder("result", model_type, farm_id, train_start_date, train_end_date, train_frequency)
    feature_path = generate_folder("result", feature_type, farm_id, train_start_date, train_end_date, train_frequency)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    log_file_path = os.path.join(model_path, "train_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        format='[%(asctime)s]-%(thread)d-%(levelname)s: %(message)s - %(filename)s:%(lineno)d')

    logging.info("{} {} {} {}".format(farm_id, train_frequency, train_start_date, train_end_date))

    # read farm_info file
    farm_info_path = '../data/farm_'+farm_id+'/farm_'+farm_id+'_info.csv'
    turbine_info = pd.read_csv(farm_info_path)

    train_farm_local(train_data_path, model_path, feature_path, turbine_info, data_resampling)



