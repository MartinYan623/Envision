# coding=utf-8
import os
import sys
sys.path.append('../')
import logging
import sys
from sklearn.externals import joblib
import pandas as pd
from datetime import datetime, date
import numpy as np
from xgb_wswp.config import train_frequency, test_start_date, test_end_date, evaluate_frequency, \
    data_path, get_train_info
from power_forecast_common.common_misc import load_data_from_pkl, generate_folder
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.evaluation_misc import evaluate_wind_speed
from plot_util.plot_revised_ws import plot_revised_wind_std, plot_revised_wind_std_improved
import pickle
from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution
from evaluate_ws_rmse import calculate_baseline_rmse
from keras.models import load_model
logger = logging.getLogger(__name__)


def generate_turbine_ws_data(model, nn_model, test_data_path, feature_file_path, flag, evaluation_periods=list(range(16, 41)),
                             train_frequency=60, delta_hour=3, evaluation_frequency="10min" ):
    """
    :param model:
    :param test_data_path:
    :param feature_file_path:
    :param evaluation_periods:
    :param train_frequency:
    :param delta_hour:
    :param evaluation_frequency:
    :return:
    """
    logger.info('Predicting model for wtg {}...'.format(model._master_id))
    x_df, y_df = load_data_from_pkl(test_data_path)
    feature_ins = WsWpFeature(train_frequency, delta_hour, model._nwp_info)
    x_df, feature_dict = feature_ins.transform(x_df)

    ws_error = {}
    if flag == True:
        revised_wd_df = model.predict(x_df, feature_dict, y_df)
        # calculate std
        # wind_error_dict = evaluate_wind_speed(x_df, y_df, revised_wd_df, evaluation_periods=evaluation_periods,
        #                                   evaluation_frequency=evaluation_frequency)
        # calculate rmse
        wind_error_dict = calculate_baseline_rmse(x_df, y_df, revised_wd_df)
        ws_error.update(wind_error_dict)
        for nwp in model._nwp_info:
            x_df[nwp + ".ws"] = revised_wd_df[nwp + ".revised_ws"]
        feature_table = pd.concat([x_df, y_df], axis=1)
        feature_table.to_pickle(feature_file_path)

    else:
        revised_wd_df = model.predict(nn_model, x_df, feature_dict, y_df)
        # select obs wind speed (3-15m/s)
        # revised_wd_df = revised_wd_df[(revised_wd_df['Y.ws_tb'] >= 3) & (revised_wd_df['Y.ws_tb'] <= 15)]
        cur_std = wind_std(np.array(revised_wd_df['Y.ws_tb']), np.array(revised_wd_df['prediction']))
        print('the std on testing data after adding linear layer is:' + str(cur_std))
        ws_error['combine.ws'] = cur_std
        # store the second layer model result
        feature_table = revised_wd_df
        feature_table.to_pickle(feature_file_path)
    return ws_error

""""
def generate_farm_ws_data(farm_id, model_path, test_data_path, feature_path, evaluate_frequency):
    farm_ins = Farm(farm_id)
    wtg_list = farm_ins.wtg_list
    result_list = []
    for n, wtg_ins in enumerate(wtg_list):
        turbine_id = wtg_ins.master_id
        # when turbine_id is "b43413c4e854432fbdad23c5778370bd", there is an except.
        cur_model_path = os.path.join(model_path, "turbine_{}.bin".format(turbine_id))
        cur_test_data_path = os.path.join(test_data_path, "turbine_{}.pkl".format(turbine_id))
        cur_feature_data_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))
        if not os.path.exists(cur_model_path) or not os.path.exists(cur_test_data_path) or os.path.exists(cur_feature_data_path):
            continue
        logging.info("Use model for turbine {}.".format(turbine_id))
        model = joblib.load(cur_model_path)
        error_dict = generate_turbine_ws_data(model, cur_test_data_path, cur_feature_data_path,
                                              evaluation_frequency=evaluate_frequency)
        error_dict.update({"turbine_id": turbine_id})
        result_list.append(error_dict)

    result_df = pd.DataFrame.from_dict(result_list)
    result_df.fillna(0.0, inplace=True)
    evaluate_result = os.path.join(feature_path, "revised_ws_error.csv")
    result_df.to_csv(evaluate_result, index=False)

    # generate the plots
    file_path = os.path.join(feature_path,
                             "farm_{}_{}_ws.png".format(farm_id, datetime.strftime(test_start_date, "%Y-%m-%d")))
    plot_revised_wind_std(result_df, "farm_{}_{}".format(farm_id, datetime.strftime(test_start_date, "%Y-%m")),
                          file_path)
"""

def generate_farm_ws_data_local(model_path, test_data_path, feature_path, evaluate_frequency, turbine_info, flag=True):

    result_list = []
    for i in range(58):
        turbine_id = turbine_info.ix[i]['master_id']
        # when turbine_id is "b43413c4e854432fbdad23c5778370bd", there is an except.
        cur_model_path = os.path.join(model_path, "turbine_{}.bin".format(turbine_id))
        cur_test_data_path = os.path.join(test_data_path, "turbine_{}.pkl".format(turbine_id))
        cur_feature_data_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))

        if not os.path.exists(cur_model_path) or not os.path.exists(cur_test_data_path) or os.path.exists(cur_feature_data_path):
            print('aaa')

        logging.info("Use model for turbine {}.".format(turbine_id))
        model = joblib.load(cur_model_path)
        # add new code
        nn_model = load_model(cur_model_path[:-4] + '.h5')

        error_dict = generate_turbine_ws_data(model, nn_model, cur_test_data_path, cur_feature_data_path, flag,
                                             evaluation_frequency=evaluate_frequency)
        error_dict.update({"turbine_id": turbine_id})
        result_list.append(error_dict)

    result_df = pd.DataFrame.from_dict(result_list)
    result_df.fillna(0.0, inplace=True)
    evaluate_result = os.path.join(feature_path, "revised_ws_error.csv")
    result_df.to_csv(evaluate_result, index=False)

    file_path = os.path.join(feature_path,
                             "farm_{}_{}_ws.png".format(farm_id, datetime.strftime(test_start_date, "%Y-%m-%d")))
    if flag == True:
        # generate the plots
        plot_revised_wind_std(result_df, "farm_{}_{}".format(farm_id, datetime.strftime(test_start_date, "%Y-%m")),
                          file_path)
    else:
        plot_revised_wind_std_improved(result_df, "farm_{}_{}".format(farm_id, datetime.strftime(test_start_date, "%Y-%m")),
                        file_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]-%(thread)d-%(levelname)s: %(message)s - %(filename)s:%(lineno)d')
    farm_id = "57f2a"

    # baseline, linear, ridge, lasso, elasticnet
    model = 'nn_new_sampling'
    model_type = 'model_revised_ws_shift_'+model+'_partial_training_resample'
    feature_type = "test_data_{}".format(model_type[6:])

    # train_start_date, train_end_date = get_train_info(farm_id)

    # for appointed training set
    train_start_date = '2017-10-04'
    train_end_date = '2018-10-24'
    test_start_date = '2018-10-25'
    test_end_date = '2018-10-31'
    train_start_date = date(*map(int, train_start_date.split('-')))
    train_end_date = date(*map(int, train_end_date.split('-')))
    test_start_date = date(*map(int, test_start_date.split('-')))
    test_end_date = date(*map(int, test_end_date.split('-')))

    test_data_path = generate_folder(data_path, "test_data_IBM_5", farm_id, test_start_date, test_end_date, train_frequency)
    model_path = generate_folder("result", model_type, farm_id, train_start_date, train_end_date, train_frequency)
    feature_path = generate_folder("result", feature_type, farm_id, test_start_date, test_end_date, train_frequency)

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # read farm_info file
    farm_info_path = '../data/farm_' + farm_id + '/farm_' + farm_id + '_info.csv'
    turbine_info = pd.read_csv(farm_info_path)

    if model == 'baseline_new_sampling222':
        flag = True
    else:
        flag = False

    generate_farm_ws_data_local(model_path, test_data_path, feature_path, evaluate_frequency, turbine_info, flag)
