from common_misc import load_data_from_pkl
from evaluation_misc import wind_std,wind_std_distribution,calculate_mbe
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
#print(data)

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
    #plt.savefig('plot/pw_ws/%s_pw_ws.png' % start_day, dpi=300)
    plt.show()


# 0-4, 4-12(every 1 one bin) and 12+
def ws_std_distribution(ws_obs, ws_predict):
    mean_bias_error = calculate_mbe(ws_obs, ws_predict)
    is_valid = ~np.isnan(ws_obs) & ~np.isnan(ws_predict)
    ws_obs_valid = ws_obs[is_valid]
    ws_predict_valid = ws_predict[is_valid]
    bins = [[0, 4]]
    for n in range(4, 12, 1):
        bins.append([n, n + 1])
    bins.append([12, 1000])
    errors = []
    for bin in bins:
        cmp_index = (ws_obs_valid >= bin[0]) & (ws_obs_valid < bin[1])
        ws_obs_eff = ws_obs_valid[cmp_index]
        if len(ws_obs_eff) == 0:
            errors.append([bin[0], 0, 0.0])
            continue
        ws_predict_eff = ws_predict_valid[cmp_index]
        cur_std = wind_std(ws_obs_eff, ws_predict_eff, mean_bias_error)
        errors.append([bin[0], len(ws_obs_eff), cur_std])
    return errors


def pw_distribution(data,probability=0.5):
    start_day = data.iloc[0]['X_basic.time']
    Y_power = data['Y.power_tb']
    if np.isnan(Y_power).sum()/289>probability:
        return None

    # discretize pw
    pw_map = [
        {'lower': 0, 'upper': 200, 'val': 0},
        {'lower': 200, 'upper': 400, 'val': 1},
        {'lower': 400, 'upper': 600, 'val': 2},
        {'lower': 600, 'upper': 800, 'val': 3},
        {'lower': 800, 'upper': 1000, 'val': 4},
        {'lower': 1000, 'upper': 1200, 'val': 5},
        {'lower': 1200, 'upper': 1400, 'val': 6},
        {'lower': 1400, 'upper': 1600, 'val': 7},
        {'lower': 1600, 'upper': 1800, 'val': 8},
        {'lower': 1800, 'upper': 2000, 'val': 9},
        {'lower': 2000, 'upper': 9999, 'val': 10},
    ]

    data=numerical_to_bin(data,'Y.power_tb',pw_map)
    Y_power = data['Y.power_tb']
    count=Y_power.value_counts()
    pw_vlaue=[]
    pw_numbers=[]
    for k, v in count.items():
        pw_vlaue.append(k)
        pw_numbers.append(v)
    sum=np.sum(pw_numbers)
    fig = plt.figure(dpi=128, figsize=(10, 6))
    fig.autofmt_xdate()
    xlabel = ['0-200', '200-400','400-600','600-800','800-1000', '1000-1200', '1200-1400',
              '1400-1600', '1600-1800', '1800-2000', '2000+']

    plt.title('Distribution of Power ' + str(start_day))
    plt.ylabel('Numbers')
    plt.xlabel('Power(kw)')
    plt.bar(pw_vlaue, pw_numbers, 0.3, label='Power Valid Numbers '+ str(sum), fc='g')
    for a, b in zip(pw_vlaue, pw_numbers):
        plt.text(a, b + 0.05, b, ha='center', va='bottom', fontsize=8)
    plt.xticks(np.arange(11), xlabel,rotation=30)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_ws_distribution(data, probability=0.5):

    start_day = data.iloc[0]['X_basic.time']
    ECO_ws = data['EC0.ws']
    GFSO_ws = data['GFS0.ws']
    WRFO_ws = data['WRF0.ws']
    Y_ws = data['Y.ws_tb']
    if (np.isnan(Y_ws).sum() / 289) > probability:
        return None
    errors_EC0 = ws_std_distribution(ECO_ws, Y_ws)
    errors_GFSO = ws_std_distribution(GFSO_ws, Y_ws)
    errors_WRFO = ws_std_distribution(WRFO_ws, Y_ws)
    errors_Y = ws_std_distribution(Y_ws,Y_ws)

    fig = plt.figure(dpi=128, figsize=(10, 6))
    fig.autofmt_xdate()
    fig.suptitle('The Distribution of Wind Speed  Start Time:'+ str(start_day))

    errors_EC0 = np.array(errors_EC0)
    errors_GFSO = np.array(errors_GFSO)
    errors_WRFO = np.array(errors_WRFO)
    errors_Y = np.array(errors_Y)
    xlabel=['0-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11','11-12','12+']

    # adjust the interval of plot
    errors_EC0[0, 0] = errors_EC0[0, 0] + 3
    errors_GFSO[0, 0] = errors_GFSO[0, 0] + 3
    errors_WRFO[0, 0] = errors_WRFO[0, 0] + 3
    errors_Y[0, 0] = errors_Y[0, 0] + 3

    ax = plt.subplot(211)
    ax.set_title('Distribution of Std')
    ax.set_ylabel('Std')
    ax.set_xlabel('Wind Speed(m/s)')
    ax.bar(errors_EC0[:, 0].tolist(), errors_EC0[:, 2].tolist(),0.2,label='EC0',fc='r')
    ax.bar((errors_GFSO[:, 0]+0.2).tolist(), errors_GFSO[:, 2].tolist(),0.2,label='GFS0',fc='b')
    ax.bar((errors_WRFO[:, 0]+0.4).tolist(), errors_WRFO[:, 2].tolist(),0.2,label='WRF0',fc='g')

    for a, b in zip(errors_EC0[:, 0].tolist(), errors_EC0[:, 2].tolist()):
        ax.text(a, b + 0.05,  round(b,2), ha='center', va='bottom', fontsize=6)
    for a, b in zip((errors_GFSO[:, 0]+0.2).tolist(), errors_GFSO[:, 2].tolist()):
        ax.text(a, b + 0.05, round(b, 2), ha='center', va='bottom', fontsize=6)
    for a, b in zip((errors_WRFO[:, 0]+0.4).tolist(), errors_WRFO[:, 2].tolist()):
        ax.text(a, b + 0.05, round(b, 2), ha='center', va='bottom', fontsize=6)

    plt.xticks(np.arange(10)+3.2, xlabel)
    plt.legend()
    plt.grid(True)

    ax = plt.subplot(212)
    ax.set_title('Distribution of Numbers')
    ax.set_ylabel('Numbers')
    ax.set_xlabel('Wind Speed(m/s)')
    ax.bar(errors_EC0[:, 0].tolist(), errors_EC0[:, 1].tolist(), 0.2, label='EC0', fc='r')
    ax.bar((errors_GFSO[:, 0] + 0.2).tolist(), errors_GFSO[:, 1].tolist(), 0.2, label='GFS0', fc='b')
    ax.bar((errors_WRFO[:, 0] + 0.4).tolist(), errors_WRFO[:, 1].tolist(), 0.2, label='WRF0', fc='g')
    ax.bar((errors_Y[:, 0] + 0.6).tolist(), errors_Y[:, 1].tolist(), 0.2, label='Y.power_tb Valid Numbers '+
                                                                                str(int(np.sum(errors_Y[:, 1].tolist()))), fc='y')
    for a, b in zip(errors_EC0[:, 0].tolist(), errors_EC0[:, 1].tolist()):
        ax.text(a, b + 0.05, int(b), ha='center', va='bottom', fontsize=6)
    for a, b in zip((errors_GFSO[:, 0] + 0.2).tolist(), errors_GFSO[:, 1].tolist()):
        ax.text(a, b + 0.05, int(b), ha='center', va='bottom', fontsize=6)
    for a, b in zip((errors_WRFO[:, 0] + 0.4).tolist(), errors_WRFO[:, 1].tolist()):
        ax.text(a, b + 0.05, int(b), ha='center', va='bottom', fontsize=6)
    for a, b in zip((errors_Y[:, 0] + 0.6).tolist(), errors_Y[:, 1].tolist()):
        ax.text(a, b + 0.05, int(b), ha='center', va='bottom', fontsize=6)
    plt.xticks(np.arange(10) + 3.2, xlabel)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

# attribute reduction
def numerical_to_bin(data, attr, val_map):
    result = data.copy(deep=True)
    for the_map in val_map:
        lower = the_map['lower']
        upper = the_map['upper']
        val = the_map['val']
        result.loc[np.logical_and(
            data[attr] >= lower, data[attr] < upper), attr] = val
    return result

#363 is the 17.06.29 set.id
# maximum id is 394
start_day=1
end_day=394
for i in range(start_day,end_day,1):
    data_1=data[data['i.set']==i]
    plot_ws(data_1)
    #plot_pw_ws(data_1)
    #plot_ws_distribution(data_1)
    #plot_pw_distribution(data_1)
