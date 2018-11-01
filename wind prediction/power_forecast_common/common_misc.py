# encoding: utf-8
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime


logger = logging.getLogger(__name__)


def generate_folder(root_folder, prefix, farm_id, start_date, end_date, frequency):
    folder_name = "{}_{}_{}_{}".format(prefix, datetime.strftime(start_date, "%Y-%m-%d"),
                                       datetime.strftime(end_date, "%Y-%m-%d"), frequency)
    return os.path.join(root_folder, "farm_{}".format(farm_id), folder_name)

def load_data_from_pkl(pkl_file):
    # need to format the timestamp column
    df = pd.read_pickle(pkl_file)
    y_label = [x for x in df.columns if x.lower().startswith('y')]
    if not bool(y_label):
        logger.warning('Y data missed.')
        x_df = df
        y_df = None
    else:
        y_df = df[y_label]
        x_df = df.drop(y_label, axis=1)
    return x_df, y_df

def forecast_linear_interpolation(pw_forecast, interval_hour, daily_hourly_sample=49):
    # assuming the sampling rate for pw_forecast is hourly, and we would like to resample it to 15 minutes
    total_days = int(len(pw_forecast) / daily_hourly_sample)
    pw_forecast_interpolate = np.array([])
    for day in range(0, total_days):
        start_index = day * daily_hourly_sample
        end_index = (day + 1) * daily_hourly_sample
        cur_array = pw_forecast[start_index: end_index]
        cur_array_interpolate = daily_forecast_linear_interpolation(cur_array, interval_hour)
        pw_forecast_interpolate = np.append(pw_forecast_interpolate, cur_array_interpolate)
    return pw_forecast_interpolate


def daily_forecast_linear_interpolation(daily_pw_forecast, interval_hour):
    # assume the length of daily_pw_forecast is 49.
    return np.interp(np.arange(0., len(daily_pw_forecast) - 1 + 1e-5, interval_hour),
                     np.arange(0., len(daily_pw_forecast) - 1 + 1e-5), daily_pw_forecast)

def forecast_nonlinear_interpolation(pw_forecast, interval_hour, daily_hourly_sample=25):
    """
    # todo this method may not be correct
    :param pw_forecast:
    :param interval_hour:
    :return:
    assuming the sampling rate for pw_forecast is 1hour and the start time is 00:00:00
    and we would like to resampling it based on interval_hour
    """
    total_days = int(len(pw_forecast) / daily_hourly_sample)
    pw_forecast_interpolate = []
    for day in range(0, total_days):
        start_index = day * daily_hourly_sample
        end_index = (day + 1) * daily_hourly_sample
        cur_array = pw_forecast[start_index: end_index]
        cur_array_interpolate = daily_nonlinear_interpolation(cur_array, interval_hour)
        pw_forecast_interpolate = np.append(pw_forecast_interpolate, cur_array_interpolate)
    return pw_forecast_interpolate


def daily_nonlinear_interpolation(daily_pw_forecast, interval_hour):
    daily_list = []
    for n in range(1, len(daily_pw_forecast)):
        start_value = daily_pw_forecast[n-1]
        end_value = daily_pw_forecast[n]
        cur_interpolation = nonlinear_mapping(start_value, end_value, interval_hour)
        if n == (len(daily_pw_forecast) - 1):
            daily_list.extend(cur_interpolation)
        else:
            daily_list.extend(cur_interpolation[:-1])
    return daily_list


def nonlinear_mapping(x1, x2, interval_hour):
    """
    :param x1: wind power at time the first hour
    :param x2: wind power at time the next hour,
    :param interval_hour: estimate the wind power interval
    :return: the estimated value in between
    """
    estimate_number = 1 / interval_hour
    ratio = np.float_power(x2/x1, 1/3.0)
    estimate_list = []
    for n in range(int(estimate_number) + 1):
        factor = n / estimate_number
        value = x1 * np.float_power(1 - factor + factor * ratio, 3)
        estimate_list.append(value)
    return estimate_list
