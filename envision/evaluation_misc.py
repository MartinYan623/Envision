# coding=utf-8
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_evaluation_index(data_df, evaluation_periods, samples_per_hour=1):
    # this evaluation period is with respect to the period from 0-49
    if len(evaluation_periods) == 49:
        return data_df.index.values

    if samples_per_hour > 1:
        evaluation_periods = list(range(evaluation_periods[0]*samples_per_hour,
                                        evaluation_periods[-1]*samples_per_hour + 1))

    evaluation_index = []
    set_list = data_df["i.set"].unique()
    for i_set in set_list:
        index = data_df[data_df["i.set"] == i_set].index[evaluation_periods]
        evaluation_index.extend(index)
    return evaluation_index


def wind_std(ws_obs, ws_predict, mean_bias_error=None):
    # for each turbine
    is_valid = ~np.isnan(ws_obs) & ~np.isnan(ws_predict)
    if mean_bias_error is None:
        mean_bias_error = calculate_mbe(ws_predict, ws_obs)
    std = np.sqrt(np.mean((ws_predict[is_valid] - ws_obs[is_valid] - mean_bias_error) ** 2))
    return std


def calculate_mbe(ws_predict, ws_obs):
    # calculate mean bias error
    is_valid = ~np.isnan(ws_obs) & ~np.isnan(ws_predict)
    ws_obs_eff = ws_obs[is_valid]
    ws_predict_eff = ws_predict[is_valid]
    mean_bias_error = np.mean(ws_predict_eff - ws_obs_eff)
    return mean_bias_error


def wind_std_distribution(ws_obs, ws_predict):
    mean_bias_error = calculate_mbe(ws_obs, ws_predict)
    is_valid = ~np.isnan(ws_obs) & ~np.isnan(ws_predict)
    ws_obs_valid = ws_obs[is_valid]
    ws_predict_valid = ws_predict[is_valid]
    bins = [[0, 4]]
    for n in range(4, 12, 2):
        bins.append([n, n + 2])
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


def evaluate_wind_speed(x_df, y_df, ws_forecast, nwp_list, evaluation_periods=list(range(49)), evaluation_frequency="60min"):
    samples_per_hour = int(60.0 / float(evaluation_frequency[:-3]))
    test_index = extract_evaluation_index(x_df[["X_basic.time", "X_basic.forecast_time", "i.set"]],
                                          evaluation_periods, samples_per_hour)
    error_dict = {}
    for nwp in nwp_list:
        col = nwp + ".ws"
        if col not in x_df.columns:
            logging.info("No original wind speed for {}".format(nwp))
        org_error = wind_std(y_df.loc[test_index, "Y.ws_tb"], x_df.loc[test_index, col])
        error_dict.update({col.split(".")[0] + "_org": org_error})

        pred_col = nwp + ".revised_ws"
        if pred_col not in x_df.columns:
            logging.info("No predicted wind speed for {}".format(nwp))
        cur_error = wind_std(y_df.loc[test_index, "Y.ws_tb"], ws_forecast.loc[test_index, pred_col])
        error_dict.update({col.split(".")[0] + "_revised": cur_error})
    return error_dict


def evaluate_wind_power(x_df, y_df, pw_forecast, evaluation_periods=list(range(49)), evaluation_frequency="60min"):
    samples_per_hour = int(60.0 / float(evaluation_frequency[:-3]))
    test_index = extract_evaluation_index(x_df[["X_basic.time", "X_basic.forecast_time", "i.set"]],
                                          evaluation_periods, samples_per_hour)
    error = calculate_rmse(y_df.loc[test_index, "Y.power_tb"], pw_forecast[test_index])
    return {"power_rmse": error}


def calculate_rmse(data1, data2):
    return np.sqrt(np.nanmean((data1 - data2) ** 2))


def xgb_ws_error(pred_array, dtrain):
    obs_array = dtrain.get_label()
    error = np.mean(obs_array * (obs_array - pred_array) ** 2)
    return "mean-square-error", error


def xgb_ws_obj(pred_array, dtrain):
    obs_array = dtrain.get_label()
    grad = - 2 * obs_array * (obs_array - pred_array)
    hess = 2 * obs_array
    return grad, hess
