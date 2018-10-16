import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy.random import seed
from common_misc import load_data_from_pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA

# 差分操作
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data

x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')

data_train = pd.concat([x_train, y_train], axis=1)

# drop out nan value
data_train = data_train.dropna(subset=['Y.ws_tb'])

# delete duplicated data
data_train = data_train.drop_duplicates(['X_basic.time'])


data_train = data_train[data_train['X_basic.time']>'2017-07-11 23:50:00']
data_train1 = data_train[data_train['X_basic.time']<'2017-07-14 00:10:00']
data_test = data_train[data_train['X_basic.time']>'2017-07-14 00:00:00']
data_test = data_test[data_test['X_basic.time']<'2017-07-14 12:00:00']

# select time and target attribute
time1 = data_train1['X_basic.time'].tolist()
time2 = data_test['X_basic.time'].tolist()
data_train1 = pd.Series(data_train1['Y.ws_tb'].tolist(), index= time1)
data_test = pd.Series(data_test['Y.ws_tb'].tolist(), index= time2)

#print(data_train1)
#print(data_test)

"""
# plot train data
plt.plot(data_train1.index, data_train1, label='train_data')
plt.legend()
plt.title('wind speed of train data')
plt.xlabel('time')
plt.ylabel('wind speed')
plt.show()


# check stable (adf test)
def adf_test(x):
    adftest = ts.adfuller(x, 1)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    print(int(adf_res['Lags Used']))
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res
x = np.array(data_train1)
print(adf_test(x))

# one order difference operation
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = data_train1.diff(1)
diff1.plot(ax=ax1, label='train_data')
plt.legend()
plt.title('wind speed of train data')
plt.xlabel('time')
plt.ylabel('wind speed')
plt.show()

# adf test of one first difference
diff_12_1 = data_train1.diff(1)
diff_12_1.dropna(inplace=True)
x = np.array(diff_12_1)
print(adf_test(x))

# white noise test
r,q,p = sm.tsa.acf(data_train1.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
"""

fig = plt.figure(figsize=(12,8))
# 时间序列的差分d,需要得到一个平稳时间序列(方差小)
# 一阶差分不行可以用二阶
# 通过自相关图和偏相关图去找寻合适的p,q
ax1 = fig.add_subplot(211)
# 自相关图决定系数q
fig = sm.graphics.tsa.plot_acf(data_train1, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
# 偏相关图决定系数p
fig = sm.graphics.tsa.plot_pacf(data_train1, lags=40, ax=ax2)
plt.show()


"""
print(sm.tsa.arma_order_select_ic(data_train1,max_ar=5,max_ma=5,ic='aic')['aic_min_order'])
print(sm.tsa.arma_order_select_ic(data_train1,max_ar=5,max_ma=5,ic='bic')['bic_min_order'])
print(sm.tsa.arma_order_select_ic(data_train1,max_ar=5,max_ma=5,ic='hqic')['hqic_min_order'])
"""

model = sm.tsa.ARMA(data_train1,(3,0)).fit()
resid = model.resid
# 对残差再做acf和pacf图
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
plt.show()

# do Ljung-Box test with resid
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_dta = model .predict('2017-07-14 00:10:00', '2017-07-14 11:50:00', dynamic=True)
print(round(predict_dta, 5))
plt.plot(data_train1.index, data_train1, label='train_data')
plt.plot(predict_dta.index, predict_dta, label='prediction')
plt.plot(data_test.index, data_test, label='test_data')
plt.legend()
plt.title('wind speed prediction')
plt.xlabel('time')
plt.ylabel('wind speed')
plt.show()