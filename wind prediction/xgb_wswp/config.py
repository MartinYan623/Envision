# coding=utf-8

from datetime import datetime

data_path = "../data"

NWP_TRAIN_BASELINE = ["EC0", "GFS0"]
NWP = {'EC': 1, 'GFS': 1, 'WRF': 1, "IBM": 1, "ENS_AVG": 1}
NWP_WS_TRAINING = [key + "0" for key in NWP.keys()]
NWP_WP_TRAINING = NWP_WS_TRAINING

#NWP_WP_TRAINING = [key + "0" for key in NWP.keys() if key != "IBM"]

train_frequency = "60min" # "10min", "60min"
evaluate_frequency = "60min" # "10min", "60min"

test_start_date = datetime(2018, 8, 1)
test_end_date = datetime(2018, 8, 30)


def get_train_info(farm_id):
    train_start_date = datetime(2017, 8, 1)
    train_end_date = datetime(2018, 7, 30)
    if farm_id == "645c1e2a0d4d44909827887d95f8c2e0":  # 麻黄滩
        train_start_date = datetime(2017, 7, 1)
        train_end_date = datetime(2018, 7, 29)
    elif farm_id in ["6c99116ce65d43d9bd32d3d75e8a12f9", "1c29d007ec00a000",
                     "57f2a7f2a624402c9565e51ba8d171cb", "WF0010", "WF0085",
                     "8c99fdbfed064805a3444fc056bbce87", "WF0016"]:
        # 切吉, # 观日台 # 鲁南 #雪邦山 #滨海 #茶卡 #头罾
        pass
    return train_start_date, train_end_date


