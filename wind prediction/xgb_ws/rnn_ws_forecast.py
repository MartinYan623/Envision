import os
import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from power_forecast_common.xgb_model import XgbForecast
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution
from xgb_ws_forecast import XgbWsForecast
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
from sklearn.linear_model import LinearRegression
from plot_util.plot_revised_ws import plot_revised_wind_std, plot_revised_wind_std_improved

length = 6
def create_lstm_model(stateful):
    model = Sequential()
    model.add(
        LSTM(20, stateful=stateful, return_sequences=False, batch_input_shape=(1, length, 1)))
    model.add(Dense(1))
    return model

def train_farm_local(train_data_path, model_path, turbine_info):

    for i in range(58):
        print(i)
        turbine_id = turbine_info.ix[i]['master_id']
        turbine_file_path = os.path.join(train_data_path, "turbine_{}.pkl".format(turbine_id))
        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))

        print("training for turbine {}".format(turbine_id))
        x_df, y_df = load_data_from_pkl(turbine_file_path)
        nwp_list = get_nwp_list(x_df.columns.values)

        for nwp in nwp_list:
            print(nwp)
            # only use ws in rnn
            nwp_ws = nwp + ".ws"

            data_input = np.array(x_df[nwp_ws])
            expected_output = list(y_df['Y.ws_tb'])
            empty = []
            for i in range(length, len(data_input) + 1):
                nan_num = np.isnan(data_input[i - length:i]).sum()
                if nan_num == 0 and ~np.isnan(expected_output[i - 1]):
                    tmp = data_input[i - length:i].tolist()
                    tmp.append(expected_output[i - 1])
                    empty.append(tmp)

            # split train data
            empty = np.array(empty)
            x_train = empty[:, :length]
            y_train = empty[:, length]

            # reshape train and test data
            num = len(x_train)
            x_train = x_train.reshape(num, length, 1)
            y_train = np.array(y_train).reshape(num, 1)
            # print(x_train.shape)
            # print(y_train.shape)

            # set parameters of rnn
            batch_size = 1
            epochs = 5

            print('Creating Stateful LSTM Model...')
            model_lstm_stateful = create_lstm_model(stateful=True)
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
            model_lstm_stateful.compile(optimizer=adam, loss='mean_squared_error')

            # train the model
            for i in range(epochs):
                print('Epoch', i + 1, '/', epochs)
                model_lstm_stateful.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=False, verbose=2)
                model_lstm_stateful.reset_states()

            # save rnn model
            model_file_path = os.path.join(model_path, "turbine_{}_{}.h5".format(turbine_id, nwp))
            model_lstm_stateful.save_weights(model_file_path)

            predicted_rnn = model_lstm_stateful.predict(x_train, batch_size=batch_size)

            if nwp == 'EC0':
                a = predicted_rnn
            else:
                a = np.concatenate((a, predicted_rnn), axis=1)

        model_linear = LinearRegression(fit_intercept=False)
        model_linear.fit(a, y_train)

        # save linear model
        model_file_path = os.path.join(model_path, "turbine_{}.pkl".format(turbine_id))
        joblib.dump(model_linear, model_file_path)

def generate_farm_ws_data_local(model_path, test_data_path, feature_path, turbine_info):
    result_list = []
    for i in range(58):
        error_dict = {}
        turbine_id = turbine_info.ix[i]['master_id']
        # when turbine_id is "b43413c4e854432fbdad23c5778370bd", there is an except.
        cur_test_data_path = os.path.join(test_data_path, "turbine_{}.pkl".format(turbine_id))

        print("testing for turbine {}".format(turbine_id))
        x_df, y_df = load_data_from_pkl(cur_test_data_path)
        nwp_list = get_nwp_list(x_df.columns.values)

        for nwp in nwp_list:
            # only use ws in rnn
            nwp_ws = nwp + ".ws"

            data_input = np.array(x_df[nwp_ws])
            expected_output = list(y_df['Y.ws_tb'])
            empty = []
            for i in range(length, len(data_input) + 1):
                nan_num = np.isnan(data_input[i - length:i]).sum()
                if nan_num == 0 and ~np.isnan(expected_output[i - 1]):
                    tmp = data_input[i - length:i].tolist()
                    tmp.append(expected_output[i - 1])
                    empty.append(tmp)

            # split test data
            empty = np.array(empty)
            x_test = empty[:, :length]
            y_test = empty[:, length]

            # reshape test data
            num = len(x_test)
            x_test = x_test.reshape(num, length, 1)
            y_test = np.array(y_test).reshape(num, 1)
            # print(x_test.shape)
            # print(y_test.shape)

            # set parameters of rnn
            batch_size = 1

            # load rnn model
            cur_model_path = os.path.join(model_path, "turbine_{}_{}.h5".format(turbine_id, nwp))

            model_lstm_stateful = create_lstm_model(stateful=True)
            model_lstm_stateful.load_weights(cur_model_path)
            predicted_rnn = model_lstm_stateful.predict(x_test, batch_size=batch_size)

            if nwp == 'EC0':
                a = predicted_rnn
            else:
                a = np.concatenate((a, predicted_rnn), axis=1)

        # load linear model
        cur_model_path = os.path.join(model_path, "turbine_{}.pkl".format(turbine_id))
        model_linear = joblib.load(cur_model_path)

        predicted_linear = model_linear.predict(a)
        revised_wd_df = pd.DataFrame({'prediction': predicted_linear.flatten(), 'Y.ws_tb': y_test.flatten()})
        # select obs wind speed (3-15m/s)
        # revised_wd_df = revised_wd_df[(revised_wd_df['Y.ws_tb'] >= 3) & (revised_wd_df['Y.ws_tb'] <= 15)]
        result = wind_std(np.array(revised_wd_df['Y.ws_tb']), np.array(revised_wd_df['prediction']))
        error_dict['combine.ws'] = result
        error_dict.update({"turbine_id": turbine_id})
        result_list.append(error_dict)

        feature_file_path = os.path.join(feature_path, "turbine_{}.pkl".format(turbine_id))
        feature_table = revised_wd_df
        feature_table.to_pickle(feature_file_path)

    result_df = pd.DataFrame.from_dict(result_list)
    result_df.fillna(0.0, inplace=True)
    evaluate_result = os.path.join(feature_path, "revised_ws_error.csv")
    result_df.to_csv(evaluate_result, index=False)

    file_path = os.path.join(feature_path,
                             "farm_{}_{}_ws.png".format(farm_id, datetime.strftime(test_start_date, "%Y-%m-%d")))

    plot_revised_wind_std_improved(result_df,
                                       "farm_{}_{}".format(farm_id, datetime.strftime(test_start_date, "%Y-%m")),
                                       file_path)
if __name__ == '__main__':

    farm_id = "57f2a"
    # train_start_date, train_end_date = get_train_info(farm_id)
    # for appointed training set
    train_start_date = '2018-08-18'
    train_end_date = '2018-10-17'
    test_start_date = '2018-10-25'
    test_end_date = '2018-10-31'
    train_start_date = date(*map(int, train_start_date.split('-')))
    train_end_date = date(*map(int, train_end_date.split('-')))
    test_start_date = date(*map(int, test_start_date.split('-')))
    test_end_date = date(*map(int, test_end_date.split('-')))

    model = 'rnn_new_sampling'
    model_type = 'model_revised_ws_shift_'+model+'_partial_training_resample'
    feature_type = "test_data_{}".format(model_type[6:])

    train_data_path = generate_folder(data_path, "train_data_IBM_5", farm_id, train_start_date, train_end_date, train_frequency)
    model_path = generate_folder("result", model_type, farm_id, train_start_date, train_end_date, train_frequency)
    feature_path = generate_folder("result", feature_type, farm_id, test_start_date, test_end_date, train_frequency)
    test_data_path = generate_folder(data_path, "test_data_IBM_5", farm_id, test_start_date, test_end_date, train_frequency)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    log_file_path = os.path.join(model_path, "train_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # read farm_info file
    farm_info_path = '../data/farm_'+farm_id+'/farm_'+farm_id+'_info.csv'
    turbine_info = pd.read_csv(farm_info_path)

    #train_farm_local(train_data_path, model_path, turbine_info)
    generate_farm_ws_data_local(model_path, test_data_path, feature_path, turbine_info)




