# coding=utf-8
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import operator

from mlApproach import util
from power_forecast_common.data_resampling import check_ws_dist, resample_data_duplicate, resample_data_gaussian
from power_forecast_common.xgb_util import hyperparameter_search, cross_prediction
from mlApproach.ml_wtg_forecast import MlWtgForecast

logger = logging.getLogger(__name__)


class XgbForecast(MlWtgForecast):
    """
    使用sklearn.ensemble包GradientBoostingRegressor工具对单风机做预报
    instance fields:
        _master_id: string 风机id
        _lat, _lon: float 风机经纬度
        _nwp_info: list 使用到的数据源种类
        _start_hour: int 每组预报第一个点的小时数
        _horizon: int 每组预报的长度
        _n_sector: int 风向离散化扇区数
        _estimator_: dict 训练好的机器学习模型，eg. {'label1': est1, 'label2': est2}
    """
    def __init__(self, master_id, lat, lon, grid_params):
        logger.info('Start XGBoost forecast for wtg %s' % master_id)
        super(XgbForecast, self).__init__(master_id, lat, lon)
        self._grid_params = grid_params
        self._n_iter = 10
        self._n_sector = 8
        self._k_fold = 5
        self._max_trees = 500
        self._error_ = {}
        self._data_resampling = False

    @classmethod
    def create_from_wtg(cls, wtg):
        return XgbForecast(wtg.master_id, wtg.lat, wtg.lon)

    def save_model(self):
        return self.__dict__

    @classmethod
    def load_model(cls, attributes: dict):
        master_id = attributes['_master_id']
        lat = attributes['_lat']
        lon = attributes['_lon']
        obj = XgbForecast(master_id, lat, lon)
        obj.__dict__.update(attributes)
        return obj

    def fit(self, *args):
        return

    def predict(self, *args):
        return

    def _xgb_feature(self, x_df, y_df, x_set):
        x_clean, y_clean, is_valid = util.feature_clean(x_df, y_df)
        x_set_clean = x_set[is_valid]
        assert y_clean.empty is False

        i_fold = util.create_cv_index(x_set_clean, self._k_fold)
        x_clean_train = x_clean.loc[i_fold != 0]
        y_clean_train = y_clean.loc[i_fold != 0]

        # check to do data resampling or not
        if self._data_resampling:
            logger.info("before resampling, data distribution is ...")
            logger.info(check_ws_dist(y_clean_train))
            x_clean_train, y_clean_train = resample_data_duplicate(x_clean_train, y_clean_train)
            logger.info("after resampling, data distribution is ...")
            logger.info(check_ws_dist(y_clean_train))

        # todo: the construaction of validation set may have some problems
        dtrain = xgb.DMatrix(x_clean_train, label=y_clean_train, missing=0)
        ddev = xgb.DMatrix(x_clean.loc[i_fold == 0], label=y_clean.loc[i_fold == 0], missing=0)
        lscore, lntree, lparam, _ = hyperparameter_search(dtrain, ddev, self._grid_params,
                                                          num_boost_round=self._max_trees)
        idx_min = np.argmin(lscore)
        tmp_result, bst = cross_prediction(lparam[idx_min], x_clean, y_clean, i_fold, lntree[idx_min], return_bst=True)
        logger.info("xgb_feature step, best xgb parameters are: ")
        logger.info([lparam[idx_min], lntree[idx_min]])
        fscore_dict = bst.get_fscore()
        fsocre_sorted = sorted(fscore_dict.items(), key=operator.itemgetter(1), reverse=True)
        logger.info(fsocre_sorted)
        result = np.nan * np.zeros(len(y_df))
        result[is_valid] = tmp_result
        return bst, pd.DataFrame({"observed": y_df, "predict": result})

    def _xgb_build(self, x_df, y_df, i_fold):
        x_clean, y_clean, is_valid = util.feature_clean(x_df, y_df)
        assert y_clean.empty == False
        i_fold = i_fold[is_valid]
        dtrain = xgb.DMatrix(x_clean.loc[i_fold != 0], y_clean.loc[i_fold != 0], missing=0)
        ddev = xgb.DMatrix(x_clean.loc[i_fold == 0], y_clean.loc[i_fold == 0], missing=0)
        lscore, lntree, lparam, _ = hyperparameter_search(dtrain, ddev, self._grid_params,
                                                          num_boost_round=self._max_trees)
        idx_min = np.argmin(lscore)
        tmp_result, bst = cross_prediction(lparam[idx_min], x_clean, y_clean, i_fold, lntree[idx_min], return_bst=True)
        result = np.nan * np.zeros(len(y_df))
        result[is_valid] = tmp_result
        return bst, result

    # def _xgb_build(self, x_df, y_df, x_set):
    #     x_clean, y_clean, is_valid = util.feature_clean(x_df, y_df)
    #     x_set_clean = x_set[is_valid]
    #     assert y_clean.empty is False
    #
    #     i_fold = util.create_cv_index(x_set_clean, self._k_fold)
    #     dtrain = xgb.DMatrix(x_clean.loc[i_fold != 0], label=y_clean.loc[i_fold != 0], missing=0)
    #     ddev = xgb.DMatrix(x_clean.loc[i_fold == 0], label=y_clean.loc[i_fold == 0], missing=0)
    #     lscore, lntree, lparam, _ = hyperparameter_search(dtrain, ddev, self._grid_params,
    #                                                       num_boost_round=self._max_trees)
    #     idx_min = np.argmin(lscore)
    #
    #     # train based on the whole dataset
    #     dtrain = xgb.DMatrix(x_clean, label=y_clean, missing=0)
    #     bst = xgb.train(lparam[idx_min], dtrain, lntree[idx_min])
    #     result = np.nan * np.zeros(len(x_df))
    #     result[is_valid] = bst.predict(dtrain)
    #     return bst, result

    def _xgb_predict(self, x_df, bst):
        result = np.nan * np.zeros(len(x_df))
        x_clean, is_valid = util.feature_clean(x_df)
        if x_clean.empty is False:
            result[is_valid] = bst.predict(xgb.DMatrix(x_clean))
        return result

    def _linear_predict(self, x_df, y_df, name, bst):
        result = bst.predict(x_df)
        data = pd.concat([x_df, y_df], axis=1)
        data = data.dropna(subset=['Y.ws_tb'])
        score = bst.score(data[name], data['Y.ws_tb'])
        return result, score


    def get_train_error(self):
        return self._error_

    def update_error(self, error_dict):
        self._error_.update(error_dict)

    def data_preprocessing(self, x_df):
        # drop all na columns
        drop_columns = []
        for col in x_df.columns:
            if np.all(np.isnan(x_df[col].values)):
                drop_columns.append(col)
        x_df.drop(drop_columns, axis=1, inplace=True)
        return x_df
