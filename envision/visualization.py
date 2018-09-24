from common_misc import load_data_from_pkl
from evaluation_misc import wind_std,wind_std_distribution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import numpy as nu
import matplotlib.dates as mdates

x_df, y_df=load_data_from_pkl('data/turbine_314e3ca4bd2345c1bc4f649f313d0b18.pkl')
#print(x_df)
#print(y_df)
# concat by column
data = pd.concat([x_df,y_df],axis=1)
print(data)


def plot_ws(data, probability=0.5):

    time=data['X_basic.time']
    start_day = data.iloc[0]['X_basic.time']
    time=time.dt.to_pydatetime()

    ECO_ws=data['EC0.ws']
    GFSO_ws=data['GFS0.ws']
    WRFO_ws=data['WRF0.ws']
    Y_ws=data['Y.ws_tb']
    if (np.isnan(Y_ws).sum()/289)>probability:
        return None
    # calculate std
    std_EC0=wind_std(ECO_ws, Y_ws, mean_bias_error=None)
    std_GFSO = wind_std(GFSO_ws, Y_ws, mean_bias_error=None)
    std_WRFO = wind_std(WRFO_ws, Y_ws, mean_bias_error=None)

    errors_EC0=wind_std_distribution(ECO_ws, Y_ws)
    errors_GFSO=wind_std_distribution(GFSO_ws, Y_ws)
    errors_WRFO=wind_std_distribution(WRFO_ws, Y_ws)

    print(errors_EC0)
    print(errors_GFSO)
    print(errors_WRFO)

    fig = plt.figure(dpi=128, figsize=(10, 6))
    fig.autofmt_xdate()
    plt.title('Wind Speed  Start Time:'+ str(start_day))
    plt.ylabel('Speed(m/s)')
    plt.xlabel('Time(min)')
    plt.grid(True)
    plt.plot(time,  ECO_ws, '-r', label='EC0.ws  std='+str(std_EC0))
    plt.plot(time,  GFSO_ws, '-b', label='GFS0.ws  std='+str(std_GFSO))
    plt.plot(time, WRFO_ws, '-y', label='WRF0.ws  std='+str(std_WRFO))
    plt.plot(time,  Y_ws, '-g', label='Y.ws_tb')

    plt.legend()
    #plt.savefig('plot/ws/%s_ws.png' % start_day, dpi=300)
    #plt.show()

def plot_pw_ws(data, probability=0.5):

    time = data['X_basic.time']
    start_day = data.iloc[0]['X_basic.time']
    time = time.dt.to_pydatetime()

    Y_power = data['Y.power_tb']
    Y_ws = data['Y.ws_tb']
    is_valid = ~np.isnan(Y_power) & ~np.isnan(Y_ws)
    # the number of both Y_power and Y_ws are valid
    # if the percentage of non-nan value is smaller than the probability, return none
    if is_valid.value_counts()[1]/289 < probability:
        return None
    Y_power_valid = Y_power[is_valid]
    Y_ws_valid = Y_ws[is_valid]
    time_valid = time[is_valid]


    fig = plt.figure(dpi=128, figsize=(10, 5))
    fig.autofmt_xdate()
    fig.suptitle('Power and Wind Speed Comparision '+str(start_day))

    ax=plt.subplot(211)
    ax.set_title('Power')
    ax.plot(time_valid, Y_power_valid, '-r', label='Y_power_valid')
    ax.set_ylabel('Power(kw)')
    ax.set_xlabel('Time(min)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    ax=plt.subplot(212)
    ax.set_title('Wind Speed')
    ax.plot(time_valid, Y_ws_valid, '-g', label='Y_ws_valid')
    ax.set_ylabel('Speed(m/s)')
    ax.set_xlabel('Time(min)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.subplots_adjust(top=0.88)
    plt.savefig('plot/pw_ws/%s_pw_ws.png' % start_day, dpi=300)
    plt.show()

"""
start_day=363  #17.06.29 set.id
for i in range(start_day,365,1):
    data_1=data[data['i.set']==i]
    plot_ws(data_1)
"""

start_day=363  #17.06.29 set.id
for i in range(start_day,394,1):
    data_1=data[data['i.set']==i]
    plot_pw_ws(data_1)
