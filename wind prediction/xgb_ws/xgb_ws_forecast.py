# coding=utf-8
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from power_forecast_common.xgb_model import XgbForecast
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution
logger = logging.getLogger(__name__)

class XgbWsForecast(XgbForecast):
    def __init__(self, master_id, lat, lon, grid_params):
        super(XgbWsForecast, self).__init__(master_id, lat, lon, grid_params)
        self._train_frequency = None
        self._data_resampling = None

    def configuration(self, train_frequency=60, k_fold=5, grid_params=None, max_trees=500,
                      data_resampling=False, seed=None):
        self._train_frequency = train_frequency
        self._k_fold = k_fold
        if seed is not None:
            np.random.seed(seed)
        if grid_params is None:
            self._grid_params = {'silent': [1], 'eta': [0.1], 'max_depth': range(3, 8), 'alpha': [0, 0.1, 0.3, 1],
                                 'min_child_weight': [0.1, 0.3, 1, 3, 10], 'subsample': [0.5, 0.6, 0.7],
                                 'lambda': [0.1, 1, 10]}
        else:
            self._grid_params = grid_params
        self._max_trees = max_trees
        self._data_resampling = data_resampling

    def fit(self, x_df, y_df, feature_dict):
        """
        :param pd.DataFrame x_df:
        :param pd.DataFrame y_df:
        :param feature_dict
        :return:
        """
        logger.info('Start fitting for wtg {}'.format(self._master_id))
        # stacking: 1st layer for wind speed
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = WsWpFeature.stacking_feature_layer1([nwp], feature_dict, single_source=True)
            x_nwp_df = self.data_preprocessing(x_df[input_feature])
            x_nwp_df, feature_names = XgbWsForecast.add_shift_features(x_nwp_df, nwp + ".ws")
            logger.info('staking 1st layer training... \n Input feature: {},\n output feature: {}'.format(
                x_nwp_df.columns.tolist(),
                output_feature))

            XgbWsForecast.wind_evaluation(y_df['Y.ws_tb'], x_nwp_df["{}.ws".format(nwp)], nwp + "_org")
            bst, result = self._xgb_feature(x_nwp_df, y_df['Y.ws_tb'], x_df['i.set'])
            self._estimator_[output_feature] = bst
            new_feature.append(pd.Series(result["predict"], index=x_df.index, name=output_feature))
            self._error_[output_feature] = {"result": result}
            XgbWsForecast.wind_evaluation(y_df['Y.ws_tb'], result["predict"], nwp)
            x_df[nwp + ".ws_predict"] = result["predict"]

        return x_df

    def predict(self, x_df, feature_dict, y_df=None):
        logger.info('Start predicting for wtg {}'.format(self._master_id))
        assert self._estimator_ != {}
        new_feature = []
        for nwp in self._nwp_info:
            input_feature, output_feature = WsWpFeature.stacking_feature_layer1([nwp], feature_dict, single_source=True)
            x_nwp_df = self.data_preprocessing(x_df[input_feature])
            x_nwp_df, feature_names = XgbWsForecast.add_shift_features(x_nwp_df, nwp + ".ws")
            revise_ws_array = self._xgb_predict(x_nwp_df, self._estimator_[output_feature])
            new_feature.append(pd.Series(revise_ws_array, index=x_df.index, name=output_feature))

        return pd.concat(new_feature, axis=1)

    @staticmethod
    def wind_evaluation(target, predict, info):
        # use predict results as ground truth
        logger.info("   {} forecast wind speed std: {}".format(info, wind_std(target, predict)))
        logger.info("   {} forecast wind std distribution is".format(info))
        logger.info(wind_std_distribution(target, predict))

    @staticmethod
    def add_shift_features(x_df, shift_col, train_frequency=60.0):
        shifted_hour_list = XgbWsForecast.get_shifted_hour_list(shift_col)
        shift_numbers = [hour * int(60.0 / train_frequency) for hour in shifted_hour_list]
        x_df, feature_names = WsWpFeature.shift_features(x_df, [shift_col], shift_numbers)
        return x_df, feature_names

    @staticmethod
    def get_shifted_hour_list(nwp_col):
        if "WRF" in nwp_col:
            shifted_hour_list = list(range(-3, 4, 1))
        else:
            shifted_hour_list = list(range(-9, 12, 3))
        shifted_hour_list.remove(0)
        return shifted_hour_list