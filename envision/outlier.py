from common_misc import load_data_from_pkl
import pandas as pd

x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')
# concat by column
data_train = pd.concat([x_train, y_train], axis=1)
data_test = pd.concat([x_test, y_test], axis=1)
count=0
for i in range(len(data_train)):
    mean_ws=(data_train.iloc[i]['EC0.ws']+data_train.iloc[i]['GFS0.ws']+data_train.iloc[i]['WRF0.ws'])/3
    if abs(mean_ws-data_train.iloc[i]['Y.ws_tb'])>3:
        count+=1
        print(count)