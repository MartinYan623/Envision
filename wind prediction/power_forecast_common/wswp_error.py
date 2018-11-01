# encoding: utf-8
import pandas as pd
import numpy as np
import logging

from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution

logger = logging.getLogger(__name__)


def check_original_std(x_df, y_df, nwp_list):
    ws_std_dict = {}
    obs_array = y_df["Y.ws_tb"].values
    for nwp in nwp_list:
        col = nwp + '.ws'
        if col not in x_df.columns:
            continue
        forecast_array = x_df[col].values
        cur_std = wind_std(obs_array, forecast_array)
        logger.info([col, cur_std])
        ws_std_dict.update({nwp: cur_std})
    # add new line
    #ws_std_dict.update({'combine.ws':  wind_std(obs_array, x_df['combine.ws'].values)})
    return ws_std_dict


def wswp_error_analysis(model, y_df):
    error_dict = model._error_
    for key, cur_error_dict in error_dict.items():
        if ".revised_ws" in key or "all.fusion_ws" in key:
            obs_array = cur_error_dict["result"]["observed"].values
            pred_array = cur_error_dict["result"]["predict"].values
            error1 = wind_std(obs_array, pred_array)
            cur_dist = wind_std_distribution(obs_array, pred_array)
            cur_error_dict.update({"error_all": error1, "error_distribution": cur_dist})
        elif "power" in key:
            error = np.sqrt(np.nanmean((cur_error_dict["result"] - y_df["Y.power_tb"]) ** 2))
            logging.info('RMSE for %s: %f' % (key, error))
            cur_error_dict.update({"error": error})


def write_wind_error(error_dictionary, wind_error_file):
    bin_list = []
    data_dict = {}
    nwp_list = []
    total_num_list = []
    for key, item in error_dictionary.items():
        if ".revised_ws" in key or "all.fusion_ws" in key:
            nwp = key.split(".")[0]
            nwp_list.append(nwp)
            dist = item["error_distribution"]
            if len(bin_list) == 0:
                bin_list = [value[0] for value in dist]
                bin_list.append("all")
                data_dict.update({"bins": bin_list})
            number_list = [value[1] for value in dist]
            number_list.append(np.sum(number_list))
            total_num_list.append(number_list[-1])
            data_dict.update({"{}_samples".format(nwp): number_list})
            error_list = [value[2] for value in dist]
            error_list.append(item["error_all"])
            data_dict.update({key: error_list})
    df = pd.DataFrame.from_dict(data_dict)
    # last_row = [""]
    # for n, nwp in enumerate(nwp_list):
    #     last_row.extend([total_num_list[n], error_dictionary[nwp]])
    # last_index = df.index[-1] + 1
    # df.loc[last_index, :] = last_row
    df.to_csv(wind_error_file, index=False)