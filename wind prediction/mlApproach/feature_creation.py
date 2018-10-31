#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongjian.liu
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

#from kong_sdk.wtg import Wtg
#from kong_sdk.weather_source import NWPSETTING
#from kong_sdk.weather_source import get_weather_forecast, get_nwp_start_time, get_weather_sequence

logger = logging.getLogger(__name__)
EXECUTOR = ThreadPoolExecutor(max_workers=20)


class ForecastFeature(object):
    """
    将风机实测风速、功率数据及天气预报数据组织成为可直接使用的预报样本
    instance fields:
        _master_id: string 风机id
        _lat, _lon: float 风机经纬度
        _forecast_time: np.ndarray([datetime]) 记录每一次预报的发布时刻
        _horizon_hour: int 每一次预报时长，以小时为单位
        _nwp_type: dict 用到的NWP预报，eg. {'EC': 2, 'GFS': 1}表示使用最新的1组GFS预报及最新的两组EC预报
        _nwp_start_time: 每一组NWP预报的开始时间
        _nwp_data: NWP预报数据
        _wind_speed: 机舱风速
        _power: 正常发电功率，已剔除限电及停机数据点
    """

    def __init__(self, master_id, lat, lon):
        logger.info('Starting to create feature table for wtg %s' % master_id)
        self._master_id = master_id
        self._lat = lat
        self._lon = lon
        self._forecast_time = None
        self._start_time = None
        self._ndays = 1
        self._horizon_hour = None
        self._nwp_type = None
        self._nwp_start_time = {}
        self._wind_speed = None
        self._power = None
        self._nwp_data = None
        self._frequency = ''

    @classmethod
    def from_wtg(cls, wtg):
        return ForecastFeature(wtg.master_id, wtg.lat, wtg.lon)

    def configuration(self, start_time, end_time=None, number_of_days=1, horizon_hour=48, nwp=None, frequency='1H'):
        """
        :param datetime start_time: 数值预报开始日期
        :param datetime end_time: 数值预报截止日期
        :param int number_of_days: 预报天数
        :param int horizon_hour: 每次预报的预报时长
        :param dict nwp: 数据源种类 eg. {'EC': 2,'GFS':1}
        :param str frequency: training data frequency
        :return: 
        """
        logger.info('Setting configuration information, wtg id: %s' % self._master_id)
        self._start_time = start_time
        if end_time is not None:
            self._ndays = (end_time - start_time).days + 1
        else:
            self._ndays = number_of_days
        logger.info('First forecast release time: %s; number of days: %s' % (start_time, self._ndays))
        self._forecast_time = pd.date_range(start_time, periods=self._ndays, freq='D')
        self._horizon_hour = horizon_hour
        self._nwp_type = nwp
        self._frequency = frequency
        info_str = ''
        for nwp, n_sets in nwp.items():
            info_str = info_str + '%i set of %s, ' % (n_sets, nwp)
        logger.info(info_str[:-2] + ' are included in feature table.')

    def get_valid_data_ratio(self):
        """返回有效的风机风速、功率数据比例"""
        if self._wind_speed is not None:
            ws_ratio = (np.count_nonzero(np.isfinite(self._wind_speed)) + 0.0) / len(self._wind_speed)
        else:
            ws_ratio = np.nan
        if self._power is not None:
            pw_ratio = (np.count_nonzero(np.isfinite(self._power)) + 0.0) / len(self._power)
        else:
            pw_ratio = np.nan
        return ws_ratio, pw_ratio

    def data_preparation(self, include_wtg_data=True):
        """根据配置信息请求风机数据（如需要）及数值天气预报数据"""
        # todo need to add the nwp data items into configure
        if include_wtg_data:
            self._get_wtg_data()
        self._get_nwp_start_time_series()
        self._nwp_data = self._get_nwp_data(('WS', 'WD', 'TMP', 'PRES'), need_density=True)

    def add_wtg_data(self, param, wtg_series):
        """传入风机风速或功率数据"""
        logging.info('Adding wtg %s data from outer source.' % param)
        if param.upper() == 'WS':
            self._wind_speed = wtg_series
        elif param.upper() == 'POWER':
            self._power = wtg_series

    @staticmethod
    def generate_key(nwp, data_key, i_set):
        if nwp == data_key:
            return "{}{}".format(nwp, i_set)
        else:
            index = data_key.rfind("_")
            return "{}{}{}".format(data_key[:index], i_set, data_key[index:])

    def create_basic_feature(self):
        # 1st edition: Y: wind_speed, power;
        #              X: (shared) time, hour, forecast_time
        #              X: (nwp) nwp_time, dist, ws, wd, tmp, pres, rho
        logger.info('Creating Y table and shared features including time, hour, forecast_time ...')
        x_time = []
        x_hour = []
        x_horizon = []
        x_forecast_time = []
        i_forecast = []
        horizon_list = list(range(self._horizon_hour + 1))
        for n, forecast_time in enumerate(self._forecast_time):
            x_forecast_time.extend([forecast_time] * len(horizon_list))
            i_forecast.extend([n] * len(horizon_list))
            time_series = [forecast_time + timedelta(hours=x) for x in horizon_list]
            x_time.extend(time_series)
            x_hour += [t.hour for t in time_series]
            x_horizon += horizon_list
        feature_table = pd.DataFrame({'X_basic.time': x_time, 'X_basic.hour': x_hour,
                                      'X_basic.forecast_time': x_forecast_time, 'X_basic.horizon': x_horizon,
                                      'i.set': i_forecast})

        # create NWP-specific features
        for nwp, n_sets in self._nwp_type.items():
            data_key_list = [key for key in self._nwp_data.keys() if nwp in key]
            for i_set in range(n_sets):
                for m, data_key in enumerate(data_key_list):
                    nwp_label = self.generate_key(nwp, data_key, i_set)
                    logger.info('Creating {} features including nwp_time, dist, wind_speed, wind_direction, density.'.format(nwp_label))
                    x_ws = []
                    x_wd = []
                    x_tmp = []
                    x_pres = []
                    x_rho = []
                    x_dist = []
                    x_nwp_time = []
                    for forecast_time, nwp_start_time in zip(self._forecast_time, self._nwp_start_time[nwp][i_set]):
                        dist = [int((forecast_time - nwp_start_time).total_seconds() / 3600) + x for x in horizon_list]
                        x_dist += dist
                        x_nwp_time += [nwp_start_time] * len(dist)
                        x_ws.append(self._nwp_data[data_key]['WS'].loc[nwp_start_time, dist].values)
                        x_wd.append(self._nwp_data[data_key]['WD'].loc[nwp_start_time, dist].values)
                        x_rho.append(self._nwp_data[data_key]['DENSITY'].loc[nwp_start_time, dist].values)
                        if nwp is not "IBM":
                            # IBM doesn't have such data
                            x_tmp.append(self._nwp_data[data_key]['TMP'].loc[nwp_start_time, dist].values)
                            x_pres.append(self._nwp_data[data_key]['PRES'].loc[nwp_start_time, dist].values)

                    if nwp is not "IBM":
                        nwp_table = pd.DataFrame({nwp_label + '.ws': np.array(x_ws).flatten(),
                                                  nwp_label + '.wd': np.array(x_wd).flatten(),
                                                  nwp_label + '.tmp': np.array(x_tmp).flatten(),
                                                  nwp_label + '.pres': np.array(x_pres).flatten(),
                                                  nwp_label + '.rho': np.array(x_rho).flatten()})
                    else:
                        nwp_table = pd.DataFrame({nwp_label + '.ws': np.array(x_ws).flatten(),
                                                  nwp_label + '.wd': np.array(x_wd).flatten(),
                                                  nwp_label + '.rho': np.array(x_rho).flatten()})
                    if m == 0:
                        nwp_table[nwp_label + '.nwp_time'] = x_nwp_time
                        nwp_table[nwp_label + '.dist'] = x_dist
                    feature_table = pd.concat([feature_table, nwp_table], axis=1)
        # create output features
        y_table_dict = {}
        if self._power is not None:
            y_table_dict['Y.power_tb'] = self._power[feature_table['X_basic.time']].values
        if self._wind_speed is not None:
            y_table_dict['Y.ws_tb'] = self._wind_speed[feature_table['X_basic.time']].values
        if y_table_dict:
            y_table = pd.DataFrame(y_table_dict)
        else:
            y_table = None
        return feature_table, y_table

    def _get_nwp_start_time_series(self):
        """基于_nwp_type信息，返回与_forecast_time相对应的模式开始时刻，记录于_nwp_start_time中"""
        for nwp, n_sets in self._nwp_type.items():
            tmp_list = []
            for ii in range(n_sets):
                nwp_start_time = get_nwp_start_time(self._start_time, nwp, ii)
                tmp_list.append(pd.date_range(nwp_start_time, periods=self._ndays, freq='D'))
            self._nwp_start_time[nwp] = tmp_list

    def _get_wtg_data(self):
        """获得与预报时段对应的风机风速及功率数据，剔除了限电及停机点"""
        logger.info('Fetching wtg wind speed and power data.')
        wtg = Wtg(self._master_id, '', self._lat, self._lon)
        start_time = self._start_time - timedelta(hours=36)
        end_time = self._start_time + timedelta(days=self._ndays, hours=self._horizon_hour)
        wind_speed = wtg.get_wtg_data(start_time, end_time, parameter='CLEAN_FILL_WIND_SPEED', freq=self._frequency)
        try:
            assert wind_speed is not None
        except AssertionError:
            logger.error('Cannot fetch wtg data.')
            raise
        active_power = wtg.get_wtg_data(start_time, end_time, parameter='CLEAN_FILL_ACTIVE_POWER', freq=self._frequency)
        wtg_status = wtg.get_wtg_status(start_time, end_time, freq=self._frequency)
        active_power[wtg_status != 0] = np.nan
        theory_power = wtg.get_theory_power_data(start_time, end_time, freq=self._frequency)
        active_power.loc[active_power.isnull()] = theory_power.loc[active_power.isnull()]
        self._wind_speed = wind_speed
        self._power = active_power
        logger.info('Valid data ratio: {}'.format(wind_speed.count() / len(active_power)))

    def _get_nwp_data(self, params_list, need_density):
        """请求NWP数据，包括风速、风向及空气密度（可选）"""
        nwp_data = {}
        for nwp in self._nwp_type.keys():
            res = NWPSETTING[nwp].RESOLUTION
            if res == 0:
                shift_latlon_list = [[0.0, 0.0]]
            else:
                shift_latlon_list = [[0.0, 0.0], [res, res], [res, -1*res], [-1*res, res], [-1*res, -1*res]]

            logger.info('Fetching %s forecast data ...' % nwp)
            nwp_start_time = get_nwp_start_time(self._start_time, nwp)
            delayed_hour = int((self._start_time - nwp_start_time).total_seconds() / 3600)

            if "IBM" in nwp:
                cur_params_list = ["WS", "WD", "DENSITY"]
            else:
                cur_params_list = params_list

            for m, shift_latlon in enumerate(shift_latlon_list):
                nwp_set = self._get_single_nwp_data(nwp, cur_params_list, nwp_start_time, delayed_hour,
                                                    shift_latlon=shift_latlon, need_density=need_density)
                if m == 0:
                    nwp_data[nwp] = nwp_set
                else:
                    nwp_data[nwp + "_shift{}".format(m)] = nwp_set
        return nwp_data

    def _get_single_nwp_data(self, nwp, cur_params_list, nwp_start_time, delayed_hour, shift_latlon=[0.0, 0.0],
                             need_density=True):
        nwp_gen = EXECUTOR.map(lambda param: self._get_nwp_data_parallel(nwp, param, nwp_start_time, delayed_hour,
                                                                         shift_latlon),
                               cur_params_list)
        nwp_set = {}
        for param, param_data in zip(cur_params_list, nwp_gen):
            nwp_set[param] = param_data

        if need_density and "DENSITY" not in nwp_set.keys():
            if 'TMP' in cur_params_list:
                t_table = nwp_set['TMP']
            else:
                t_table = self._get_nwp_data_parallel(nwp, 'TMP', nwp_start_time, delayed_hour)
            if 'PRES' in cur_params_list:
                pres_table = nwp_set['PRES']
            else:
                pres_table = self._get_nwp_data_parallel(nwp, 'PRES', nwp_start_time, delayed_hour)
            nwp_set['DENSITY'] = pres_table * 1e2 / (287.05 * (t_table + 273.15))
        return nwp_set

    def _get_nwp_data_parallel(self, nwp, param, nwp_start_time, offset, shift_latlon=[0.0, 0.0]):
        length = self._horizon_hour + 1
        columns = list(range(offset, length + offset))
        ti = pd.date_range(nwp_start_time, periods=self._ndays, freq='D')
        data_mat = get_weather_sequence(self._lat + shift_latlon[0], self._lon + shift_latlon[1], nwp, param,
                                        nwp_start_time, self._ndays, hours=length, offset=offset)
        return pd.DataFrame(data_mat, index=ti, columns=columns)

