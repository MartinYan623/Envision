#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongjian.liu
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from datetime import datetime
import logging

from .feature_creation import ForecastFeature
from . import util

logger = logging.getLogger(__name__)


class MlWtgForecast(object, metaclass=ABCMeta):
    """
    抽象类：使用机器学习对单风机做预报
    instance fields:
        _master_id: string 风机id
        _lat, _lon: float 风机经纬度
        _nwp_info: list 使用到的数据源种类
        _start_hour: int 每组预报第一个点的小时时刻
        _horizon: int 每组预报的长度
        _estimator_: dict 训练好的机器学习模型，eg. {'label1': est1, 'label2': est2}
    """

    def __init__(self, master_id, lat, lon):
        # logging.info('Start GBR forecast for wtg %s' % master_id)
        self._master_id = master_id
        self._lat = lat
        self._lon = lon
        self._estimator_ = {}
        self._nwp_info = None
        self._start_hour = None
        self._horizon = None

    def load_feature_from_pkl(self, prefix):
        """从pkl文件载入风机特征数据"""
        x_df, y_df = util.load_pkl(self._master_id, prefix)
        self._nwp_info = util.get_nwp_list(x_df.columns)
        logger.debug('{} are used to build forecast model.'.format(self._nwp_info))
        self._start_hour = x_df['X_basic.forecast_time'][0].hour
        i_set = x_df['i.set'].values
        self._horizon = len(i_set) / len(np.unique(i_set))
        return x_df, y_df

    def load_feature_from_api(self, start_time, end_time=None, number_of_days=1, nwp=None, horizon_hour=48,
                              include_wtg_data=True, frequency='1H'):
        """
        :param datetime start_time: 数值预报开始日期
        :param datetime end_time: 数值预报截止日期
        :param int number_of_days: 预报天数
        :param int horizon_hour: 每次预报的预报时长
        :param dict nwp: 数据源种类 eg. {'EC': 2,'GFS':1}
        :return:
        自API接口导入特征数据及风机数据
        """
        if self._horizon is None:
            self._horizon = horizon_hour
        else:
            horizon_hour = self._horizon
        if self._nwp_info is not None and nwp is None:
            nwp = self._get_nwp_dict()
        if self._start_hour is not None:
            start_time = start_time.replace(hour=self._start_hour, minute=0, second=0, microsecond=0)
        else:
            self._start_hour = start_time.hour
        sample = ForecastFeature(self._master_id, self._lat, self._lon)
        if end_time is None:
            sample.configuration(start_time, number_of_days=number_of_days, nwp=nwp, horizon_hour=horizon_hour,
                                 frequency=frequency)
        else:
            sample.configuration(start_time, end_time=end_time, nwp=nwp, horizon_hour=horizon_hour, frequency=frequency)
        sample.data_preparation(include_wtg_data=include_wtg_data)
        x_df, y_df = sample.create_basic_feature()
        if self._nwp_info is None:
            self._nwp_info = util.get_nwp_list(x_df.columns)
        return x_df, y_df

    def _get_nwp_dict(self):
        nwp_dict = {}
        for nwp in self._nwp_info:
            nwp_dict[nwp[:-1]] = max(int(nwp[-1]) + 1, nwp_dict.get(nwp[:-1], 0))
        return nwp_dict

    def set_nwp_list(self, nwp_list):
        self._nwp_info = nwp_list

    def clean(self):
        self._nwp_info = None
        self._estimator_ = {}
        self._start_hour = None
        self._horizon = None

    @staticmethod
    def get_rmse(data1, data2, label):
        rmse = np.sqrt(np.nanmean((data1 - data2) ** 2))
        logging.info('RMSE for %s: %f' % (label, rmse))
        return rmse

    @abstractmethod
    def data_preprocessing(self, *args):
        return

    @abstractmethod
    def fit(self, *args):
        return

    @abstractmethod
    def predict(self, *args):
        return
