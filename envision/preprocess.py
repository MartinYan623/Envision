from common_misc import load_data_from_pkl
import pandas as pd
import numpy as np
from outlier import delete_outlier_ws
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

# attribute reduction
def numerical_to_bin(data, attr, val_map):
    result = data.copy(deep=True)
    for the_map in val_map:
        lower = the_map['lower']
        upper = the_map['upper']
        val = the_map['val']
        result.loc[np.logical_and(data[attr] >= lower, data[attr] < upper), attr] = val
    return result

def substitute(data, attr, val_map):
    result = data.copy(deep=True)
    for key in val_map:
        result.loc[np.array(data[attr] == key).astype(
            np.bool), attr] = val_map[key]
    return result

def smooth_Y(data):
    smooth_y = []
    for i in range(394):
        sub_data = data[data['i.set'] == i]
        a = np.array(sub_data['Y.ws_tb'].reshape(289, 1))
        new_a = [a[0][0], a[1][0], a[2][0]]
        for j in range(3,len(a)-2):
            if np.isnan(a[j]) == False:
                # 'nanmean' skips nan value
                new_a.append(np.nanmean(a[j-3:j+3, :]))
            else:
                new_a.append(np.nan)
        new_a.append(a[287][0])
        new_a.append(a[288][0])
        smooth_y = smooth_y+new_a
    return smooth_y

# discretize wd
wd_map = [
        {'lower': 0, 'upper': 60, 'val': int(0)},
        {'lower': 60, 'upper': 120, 'val': int(1)},
        {'lower': 120, 'upper': 180, 'val': int(2)},
        {'lower': 180, 'upper': 240, 'val': int(3)},
        {'lower': 240, 'upper': 300, 'val': int(4)},
        {'lower': 300, 'upper': 360, 'val': int(5)},
    ]

season_dict = {
    '01': 'winter',
    '02': 'winter',
    '03': 'spring',
    '04': 'spring',
    '05': 'spring',
    '06': 'summer',
    '07': 'summer',
    '08': 'summer',
    '09': 'autumn',
    '10': 'autumn',
    '11': 'autumn',
    '12': 'winter'
}

time_dict={
    21:'night', 22:'night', 23:'night', 0:'night', 1:'night', 2:'night', 3:'night', 4:'night',
    5:'morning', 6:'morning', 7:'morning', 8:'morning', 9:'morning', 10:'morning',
    11:'morning', 12:'morning',
    13:'afternoon', 14:'afternoon', 15:'afternoon', 16:'afternoon', 17:'afternoon',
    18:'afternoon', 19:'afternoon', 20:'afternoon'

}

tem_map = [
        {'lower':-99, 'upper': -10, 'val': int(0)},
        {'lower':-10, 'upper': 0, 'val': int(1)},
        {'lower': 0, 'upper': 10, 'val': int(2)},
        {'lower': 10, 'upper': 20, 'val': int(3)},
        {'lower': 20, 'upper': 30, 'val': int(4)},
        {'lower': 30, 'upper': 99, 'val': int(5)},
    ]

def normalization(data,attribute):
    data[attribute]= StandardScaler().fit_transform(data[[attribute]]).reshape(-1).tolist()
    return data

def binarization(data,attribute):
    data[attribute]=Binarizer(threshold=1).fit_transform(data[[attribute]]).reshape(-1).tolist()
    return data

x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')


def preprocess(x_train,y_train,x_test,y_test):

    # concat by column
    data_train = pd.concat([x_train, y_train], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)

    # whether smooth data_y
    # data_train['Y.ws_tb'] = smooth_Y(data_train)

    # drop out nan value
    data_train = data_train.dropna(subset=['Y.ws_tb'])
    data_train = data_train[np.isnan(data_train['GFS0.ws']) == False]
    data_train = data_train[np.isnan(data_train['WRF0.ws']) == False]
    data_test = data_test.dropna(subset=['Y.ws_tb'])

    # deleter outlier
    # delete_outlier_ws(data_train)

    length_data_train = len(data_train)

    # concat train and test data
    data = pd.concat([data_train, data_test], axis=0)
    """
    # discretize wd and one-hot
    data = numerical_to_bin(data, 'GFS0.wd', wd_map)
    EC0wd_dummies = pd.get_dummies(data['GFS0.wd'], prefix='GFS0.wd')
    data = pd.concat([data,EC0wd_dummies], axis=1)
    data.drop('GFS0.wd', axis=1, inplace=True)

    # discretize temp and one-hot
    data = numerical_to_bin(data, 'EC0.tmp', tem_map)
    EC0tmp_dummies = pd.get_dummies(data['EC0.tmp'], prefix='EC0.tmp')
    data = pd.concat([data, EC0tmp_dummies], axis=1)
    data.drop('EC0.tmp', axis=1, inplace=True)

    # extract month from time and discretize month into season
    month = np.array(data['X_basic.forecast_time'].astype(str))
    data['month'] = list(map(lambda x: x.split('-')[1], month))
    data['season']=data['month']
    data = substitute(data, 'season', season_dict)
    season_dummies = pd.get_dummies(data['season'], prefix='season')
    data = pd.concat([data, season_dummies], axis=1)
    data.drop('season', axis=1, inplace=True)

    # extract month and one-hot encoding
    month_dummies = pd.get_dummies(data['month'], prefix='month')
    data = pd.concat([data, month_dummies], axis=1)

    # extract time and discretize into morning, afternoon and night
    data = substitute(data, 'X_basic.hour', time_dict)
    time_dummies = pd.get_dummies(data['X_basic.hour'], prefix='time')
    data = pd.concat([data, time_dummies], axis=1)

    #normalization
    #data = normalization(data,'EC0.ws')
    data = normalization(data, 'EC0.pres')
    data = normalization(data, 'EC0.rho')
    #data = normalization(data, 'GFS0.ws')
    data = normalization(data, 'GFS0.pres')
    data = normalization(data, 'GFS0.rho')
    #data = normalization(data, 'WRF0.ws')
    data = normalization(data, 'WRF0.pres')
    data = normalization(data, 'WRF0.rho')

    #data = binarization(data, 'EC0.rho')
    #data = binarization(data, 'GFS0.rho')
    #data = binarization(data, 'WRF0.rho')
    """

    # split train and test data and return
    train = data.iloc[:length_data_train]
    test = data.iloc[length_data_train:]
    return train, test



