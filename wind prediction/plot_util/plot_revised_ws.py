# coding=utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from xgb_wswp.config import test_start_date, test_end_date
from power_forecast_common.common_misc import generate_folder


def plot_revised_wind_std(data_df, title_str, file_path):

    nwp_list = set()
    for col in data_df.columns:
        if "turbine" in col:
            continue
        nwp_list.add(col[:col.rfind("_")])

    fig = plt.figure(dpi=128, figsize=(10, 6))
    fig.autofmt_xdate()

    rows = len(nwp_list)
    index_list = list(range(len(data_df)))
    nwp_list = sorted(nwp_list)
    for n, nwp in enumerate(nwp_list):
        ax = plt.subplot(rows, 1, 1+n)
        org_col = nwp + '_org'
        ax.plot(index_list, data_df[org_col], "-b", label="{}_org: {:.3f}".format(nwp, np.mean(data_df[org_col])))

        revised_col = nwp + "_revised"
        ax.plot(index_list, data_df[revised_col], "-r", label="{}_revised: {:.3f}".format(nwp, np.mean(data_df[revised_col])))
        if n == len(nwp_list) - 1:
            ax.set_xlabel("turbine index")
        if n == 0:
            ax.set_title(title_str)
        ax.set_ylabel("Wind Std")
        ax.grid(True)
        ax.legend(loc="upper center", ncol=2, fontsize="small", frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(file_path, dpi=300)

def plot_revised_wind_std_improved(data_df, title_str, file_path):
    std = []
    turbine_id = []
    for i in range(len(data_df)):
        std.append(data_df.ix[i]['combine.ws'])
        turbine_id.append(i+1)
    print(std)
    # calculate mean std
    mean_std = np.mean(np.array(std))
    plt.title(title_str)
    plt.plot(turbine_id[:28], std[:28], label='combine.ws:' + str(mean_std), color='r')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path, dpi=300)

def main():

    farm_id = "1c29d007ec00a000"
    train_frequency = "10min"

    test_data_path = generate_folder("result", "test_data_revised_ws", farm_id, test_start_date, test_end_date, train_frequency)
    test_file = "revised_ws_error.csv"
    test_file_path = os.path.join(test_data_path, test_file)

    data_df = pd.read_csv(test_file_path)

    file_path = os.path.join(test_data_path,
                             "farm_{}_{}_ws.png".format(farm_id, datetime.strftime(test_start_date, "%Y-%m-%d")))
    plot_revised_wind_std(data_df, "farm_{}_{}".format(farm_id, datetime.strftime(test_start_date, "%Y-%m")),
                          file_path)


if __name__ == "__main__":
    main()