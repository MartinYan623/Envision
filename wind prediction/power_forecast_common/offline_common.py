# coding=utf-8
import logging
from datetime import datetime, timedelta
from mlApproach.util import get_nwp_list


def generate_logger(log_file_path):
    file_hander = logging.FileHandler(log_file_path, "a")
    file_hander.setFormatter(
        logging.Formatter('[%(asctime)s]-%(thread)d-%(levelname)s: %(message)s - %(filename)s:%(lineno)d'))
    logger = logging.getLogger()
    logger.handlers = [file_hander]


def filter_data(x_df, y_df, end_date, start_date=None):
    if start_date is None:
        use_flag = x_df["X_basic.time"] < end_date
    else:
        use_flag = (x_df["X_basic.time"] < end_date) & (x_df["X_basic.time"] >= start_date)
    set_list = x_df[use_flag]["i.set"].unique()
    revised_flag = x_df["i.set"].isin(set_list)
    return x_df[revised_flag].reset_index(drop=True), y_df[revised_flag].reset_index(drop=True)


def split_data(x_df, y_df, split_date):
    train_flag = x_df["X_basic.time"] <= split_date
    train_set_list = x_df[train_flag]["i.set"].unique()
    train_revised_flag = x_df["i.set"].isin(train_set_list)

    train_x_df = x_df[train_revised_flag].reset_index(drop=True)
    train_y_df = y_df[train_revised_flag].reset_index(drop=True)

    test_x_df = x_df[~train_revised_flag].reset_index(drop=True)
    test_y_df = y_df[~train_revised_flag].reset_index(drop=True)
    return train_x_df, train_y_df, test_x_df, test_y_df


def generate_train_test_date(train_init_date, evaluate_days=7, evaluate_round=2):
    date_info = []
    for n in range(1):
        train_end_date = train_init_date + timedelta(days=evaluate_days) * n
        cur_info = {"train_end_date": train_end_date, "test_date": []}
        for m in range(evaluate_round):
            test_start_date = train_end_date + timedelta(days=evaluate_days) * m
            test_end_date = test_start_date + timedelta(days=evaluate_days)
            if test_end_date > datetime.now():
                return date_info
            cur_info["test_date"].append([m, test_start_date, test_end_date])
        date_info.append(cur_info)
    return date_info


def calculate_nwp_correlation(x_df):
    nwp_list = get_nwp_list(x_df.columns)
    nwp_cols = [nwp + ".ws" for nwp in nwp_list]
    corr_df = x_df[nwp_cols].corr()
    print(corr_df)