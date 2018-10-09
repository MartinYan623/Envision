from common_misc import load_data_from_pkl
import pandas as pd
import numpy as np

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

#x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
#x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')

def preprocess(data):

    # discretize wd and one-hot
    data = numerical_to_bin(data, 'GFS0.wd', wd_map)
    EC0wd_dummies = pd.get_dummies(data['GFS0.wd'], prefix='GFS0.wd')
    data = pd.concat([data,EC0wd_dummies], axis=1)
    data.drop('GFS0.wd', axis=1, inplace=True)

    # extract month from time and discretize month into season
    time=np.array(data['X_basic.forecast_time'].astype(str))
    data['month']=list(map(lambda x: x.split('-')[1],time))
    data= substitute(data, 'month', season_dict)
    season_dummies = pd.get_dummies(data['month'], prefix='season')
    data = pd.concat([data, season_dummies], axis=1)
    data.drop('month', axis=1, inplace=True)

    # split train and test data and return
    train = data.iloc[:113866]
    test = data.iloc[113866:]
    return train,test
