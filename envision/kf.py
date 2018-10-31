import numpy as np
from filterpy.gh import GHFilter
from common_misc import load_data_from_pkl
from evaluation_misc import wind_std,wind_std_distribution,calculate_mbe
import pandas as pd
import matplotlib.pyplot as plt

# g-h filter sample
def g_h_filter(data, x0, dx, g, h , dt=1., pred=None):
    x = x0
    results = []
    for z in data:
        x_est = x + (dx * dt)
        dx = dx
        if pred is not None:
            pred.append(x_est)
        residual = z - x_est
        dx = dx + h * (residual) / dt
        x = x_est + g * residual
        results.append(x)
    return np.array(results)

weights = np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
data = g_h_filter(data=weights, x0=160, dx=1, g=3./10, h=1./3, dt=1.)
#print(data)

# use the built-in function
f = GHFilter(x=0., dx=0., dt=1., g=.8, h=.2)
f.update(z=1.2)
#print(f.update(z=2.1, g=.85, h=.15))
#print(f.batch_filter([3., 4., 5.]))


x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')
data_test = pd.concat([x_test, y_test], axis=1)
data_test = data_test.dropna(subset=['Y.ws_tb'])
data_test = data_test.drop_duplicates(['X_basic.time'])
data_test = data_test.reset_index(drop=True)
data_test = data_test[data_test['X_basic.time'] < '2018-08-03 00:10:00']
data=np.array(data_test['Y.ws_tb'].tolist())
prediction = g_h_filter(data=data, x0=12.8597, dx=0.5, g=3./10, h=1./3, dt=1.)
print(prediction)
print(wind_std(data, prediction, mean_bias_error=None))

time=np.array(np.arange(289))
plt.plot(time, data, label='test_data')
plt.plot(time, prediction, label='prediction')
plt.legend()
plt.title('wind speed prediction')
plt.xlabel('time')
plt.ylabel('wind speed')
plt.show()