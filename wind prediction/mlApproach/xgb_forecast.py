#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongjian.liu
@date:   2017/11/15
"""
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

#from kong_sdk.zipkin_client.ZipkinClient import zipkin_client

from . import util
from .ml_wtg_forecast import MlWtgForecast
from .xgb_util import cross_prediction, feature_clean, get_self_or_values, hyperparameter_search

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

    def __init__(self, master_id, lat, lon):
        logger.info('Start XGBoost forecast for wtg %s' % master_id)
        super(XgbForecast, self).__init__(master_id, lat, lon)
        self._n_sector = None
        self._k_fold = None
        self._n_iter = None
        self._grid_params = None
        self._max_trees = 300

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

    def configuration(self, n_sector=8, k_fold=5, n_iter=100, grid_params=None, seed=None):
        self._n_iter = n_iter
        self._n_sector = n_sector
        self._k_fold = k_fold
        if seed is not None:
            np.random.seed(seed)
        if grid_params is None:
            self._grid_params = {'silent': [1], 'eta': [0.1], 'max_depth': range(3, 8), 'alpha': [0, 0.1, 0.3, 1],
                                 'min_child_weight': [0.1, 0.3, 1, 3, 10], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                 'lambda': [0.1, 0.3, 1, 3, 10], 'tree_method': ['exact', 'approx']}
        else:
            self._grid_params = grid_params

    def data_preprocessing(self, x_df):
        x_df = util.add_feature_wdcut(x_df, n_sector=self._n_sector, one_hot_encoding=True)
        x_df = util.add_feature_rho_crossed(x_df)
        return x_df

    #@zipkin_client()
    def fit(self, x_df, y_df):
        """
        :param pd.DataFrame x_df:
        :param pd.DataFrame y_df:
        :return:
        """
        logger.info('Start fitting for wtg {}'.format(self._master_id))
        i_fold = util.create_cv_index(x_df['i.set'], self._k_fold)
        # stacking: 1st layer
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = self._1st_stacking_feature(nwp)
            logger.info('staking 1st layer training... Input feature: {}, output feature: {}'.format(input_feature,
                                                                                                      output_feature))
            result, bst = self._build_xgb(x_df[input_feature], y_df['Y.power_tb'], i_fold)
            self._estimator_[output_feature] = bst
            new_feature.append(pd.Series(result, index=x_df.index, name=output_feature))
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # shift meta result
        x_df = self._shift_metaresult(x_df)
        # stacking: 2nd layer
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = self._2nd_stacking_feature(nwp)
            logger.info('staking 2nd layer training... Input feature: {}, output feature: {}'.format(input_feature,
                                                                                                      output_feature))
            result, bst = self._build_xgb(x_df[input_feature], y_df['Y.power_tb'], i_fold)
            self._estimator_[output_feature] = bst
            new_feature.append(pd.Series(result, index=x_df.index, name=output_feature))
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # linear regression
        logger.info('Linear regression training...')
        input_feature, output_feature = self._final_fusion_feature()
        result = self._build_linear_regression(x_df[input_feature], y_df['Y.power_tb'], x_df['X_basic.horizon'])
        return pd.Series(result, index=x_df.index, name=output_feature)

    #@zipkin_client()
    def predict(self, x_df, y_df=None):
        """
        :param pd.DataFrame x_df:
        :param pd.DataFrame y_df:
        :return:
        """
        logger.debug('Start predicting for wtg {}'.format(self._master_id))
        assert self._estimator_ != {}
        # stacking: 1st layer
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = self._1st_stacking_feature(nwp)
            logger.debug('staking 1st layer predicting... Input feature: {}, output feature: {}'.format(input_feature,
                                                                                                        output_feature))
            result = self._xgb_predict(x_df[input_feature], self._estimator_[output_feature])
            new_feature.append(pd.Series(result, index=x_df.index, name=output_feature))
            if y_df is not None:
                self.get_rmse(new_feature[-1], y_df, output_feature)
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # shift meta result
        x_df = self._shift_metaresult(x_df)
        # stacking: 2nd layer
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = self._2nd_stacking_feature(nwp)
            logger.debug('staking 2nd layer predicting... Input feature: {}, output feature: {}'.format(input_feature,
                                                                                                        output_feature))
            result = self._xgb_predict(x_df[input_feature], self._estimator_[output_feature])
            new_feature.append(pd.Series(result, index=x_df.index, name=output_feature))
            if y_df is not None:
                self.get_rmse(new_feature[-1], y_df, output_feature)
        x_df = pd.concat([x_df, *new_feature], axis=1)
        # linear regression
        logger.debug('Linear regression predicting...')
        input_feature, output_feature = self._final_fusion_feature()
        result = self._linear_regression_predict(x_df[input_feature], x_df['X_basic.horizon'])
        return result

    #@zipkin_client()
    def _build_linear_regression(self, x_df, y_df, horizon_arr):
        if x_df.shape[1] == 1:
            return x_df.values.ravel()
        _, _, is_valid = feature_clean(x_df, y_df)
        horizon_list = horizon_arr.unique()
        model_dict = {}
        result = np.zeros(len(y_df)) * np.nan
        for horizon in horizon_list:
            linear_model = Ridge(fit_intercept=False)
            linear_model.fit(x_df.loc[is_valid & (horizon_arr == horizon)], y_df[is_valid & (horizon_arr == horizon)])
            model_dict[horizon] = linear_model
            result[is_valid & (horizon_arr == horizon)] = linear_model.predict(
                x_df.loc[is_valid & (horizon_arr == horizon)])
        self._estimator_['fusion'] = model_dict
        return result

    def _build_xgb(self, x_df, y_df, i_fold, use_column_name=False):
        x_clean, y_clean, is_valid = feature_clean(x_df, y_df)
        assert y_clean.empty is False
        i_fold = i_fold[is_valid]
        dtrain = xgb.DMatrix(x_clean.loc[i_fold != 0], y_clean.loc[i_fold != 0], missing=0)
        ddev = xgb.DMatrix(x_clean.loc[i_fold == 0], y_clean.loc[i_fold == 0], missing=0)
        lscore, lntree, lparam, _ = hyperparameter_search(dtrain, ddev, self._grid_params, n_iter=self._n_iter,
                                                          verbose_eval=False, early_stopping_rounds=8,
                                                          num_boost_round=self._max_trees)
        idx_min = np.argmin(lscore)
        tmp_result, bst = cross_prediction(lparam[idx_min], x_clean, y_clean, i_fold, lntree[idx_min], return_bst=True,
                                           use_column_name=use_column_name)
        result = np.nan * np.zeros(len(y_df))
        result[is_valid] = tmp_result
        return result, bst

    def _xgb_predict(self, x_df, bst, use_column_names=False):
        result = np.nan * np.zeros(len(x_df))
        x_clean, is_valid = feature_clean(x_df)
        if x_clean.empty is False:
            result[is_valid] = bst.predict(xgb.DMatrix(get_self_or_values(x_clean, use_column_names)))
        return result

    def _linear_regression_predict(self, x_df, horizon_arr):
        if x_df.shape[1] == 1:
            return x_df.values.ravel()
        _, is_valid = feature_clean(x_df)
        horizon_list = horizon_arr.unique()
        result = np.zeros(len(x_df)) * np.nan
        for horizon in horizon_list:
            model = self._estimator_['fusion'][horizon]
            try:
                result[is_valid & (horizon_arr == horizon)] = model.predict(
                    x_df.loc[is_valid & (horizon_arr == horizon)])
            except:
                pass
        return result

    def _shift_metaresult(self, df):
        logger.debug('Shifting rho_ws_wd feature...')
        shift_feature = []
        for nwp in self._nwp_info:
            for shift_index in range(1, 4):
                shift_feature.append(util.add_feature_shift(df, nwp + '.rho_ws_wd', shift_index, concat=False))
                shift_feature.append(util.add_feature_shift(df, nwp + '.rho_ws_wd', -shift_index, concat=False))
        return pd.concat([df, *shift_feature], axis=1)

    def _1st_stacking_feature(self, nwp):
        input_feature = [nwp + '.rho_ws', nwp + '.rho_ws2', nwp + '.rho_ws3']
        input_feature.extend(['{}.wd_cut_s{}'.format(nwp, n) for n in range(self._n_sector)])
        output_feature = nwp + '.rho_ws_wd'
        return input_feature, output_feature

    @staticmethod
    def _2nd_stacking_feature(nwp):
        input_feature = [nwp + '.ws', nwp + '.dist', nwp + '.rho_ws_wd', nwp + '.rho_ws_wd_p1', nwp + '.rho_ws_wd_p2',
                         nwp + '.rho_ws_wd_p3', nwp + '.rho_ws_wd_n1', nwp + '.rho_ws_wd_n2', nwp + '.rho_ws_wd_n3']
        output_feature = nwp + '.wp_fcst'
        return input_feature, output_feature

    def _final_fusion_feature(self):
        input_feature = [nwp + '.wp_fcst' for nwp in self._nwp_info]
        output_feature = 'final_fcst'
        return input_feature, output_feature
