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

x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')

data_train = pd.concat([x_train, y_train], axis=1)
#data_test = pd.concat([x_test, y_test], axis=1)

# drop out nan value
data_train = data_train.dropna(subset=['Y.ws_tb'])
#data_test = data_test.dropna(subset=['Y.ws_tb'])

# delete duplicated data
data_train = data_train.drop_duplicates(['X_basic.time'])
#data_test = data_test.drop_duplicates(['X_basic.time'])


data_train1 = data_train[data_train['X_basic.time']<'2017-07-12 00:00:00']
data_test = data_train[data_train['X_basic.time']>'2017-07-11 23:50:00']
data_test = data_test[data_test['X_basic.time']<'2017-07-12 03:00:00']
"""
# select time and target attribute
time1 = data_train1['X_basic.time'].tolist()
time2 = data_test['X_basic.time'].tolist()
data_train1 = pd.Series(data_train1['Y.ws_tb'].tolist(), index= time1)
data_test = pd.Series(data_test['Y.ws_tb'].tolist(), index= time2)
"""
data_train1.index = pd.Index(sm.tsa.datetools.dates_from_range('1900','2090'))
fig = plt.figure(figsize=(12, 8))
data_train1=data_train1['Y.ws_tb']
print(data_test['Y.ws_tb'])


# 时间序列的差分d,需要得到一个平稳时间序列(方差小)
# 一阶差分不行可以用二阶
diff1 = data_train1.diff(1)
# 通过自相关图和偏相关图去找寻合适的p,q
ax1 = fig.add_subplot(211)
# 自相关图决定系数q
fig = sm.graphics.tsa.plot_acf(data_train1, lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
# 偏相关图决定系数p
fig = sm.graphics.tsa.plot_pacf(data_train1, lags=20, ax=ax2)
plt.show()


x = np.array(data_train1.tolist())
def adf_test(x):
    adftest = ts.adfuller(x, 1)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    print(int(adf_res['Lags Used']))
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res
print(adf_test(x))

model = sm.tsa.ARMA(data_train1,(1,0)).fit()
predict_dta = model.predict('2090', '2108', dynamic=True)
print(round(predict_dta,5))