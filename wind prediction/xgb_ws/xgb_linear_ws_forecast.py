# coding=utf-8
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from power_forecast_common.xgb_model import XgbForecast
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution
from xgb_ws_forecast import XgbWsForecast

logger = logging.getLogger(__name__)


class XgbLinearWsForecast(XgbWsForecast):

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

        new_data = pd.concat([x_df, y_df], axis=1)
        name = []
        for nwp in self._nwp_info:
            name.append(nwp + ".ws_predict")
            new_data = new_data.dropna(subset=[nwp + ".ws_predict"])

        new_data = new_data.dropna(subset=['Y.ws_tb'])

        # non horizon
        lr = LinearRegression()
        combine = lr.fit(new_data[name], new_data['Y.ws_tb'])
        self._estimator_['combine.ws'] = combine

        # # add new horizon
        # horizon_list = new_data['X_basic.horizon'].unique()
        # model_dict = {}
        # for horizon in horizon_list:
        #     lr = LinearRegression(fit_intercept=False)
        #     lr.fit(new_data[new_data['X_basic.horizon'] == horizon][name],
        #            new_data[new_data['X_basic.horizon'] == horizon]['Y.ws_tb'])
        #     model_dict[horizon] = lr
        # self._estimator_['combine.ws'] = model_dict

        return x_df

    def predict(self, x_df, feature_dict, y_df=None):
        logger.info('Start predicting for wtg {}'.format(self._master_id))
        assert self._estimator_ != {}
        new_feature = []
        result = pd.DataFrame({})
        for nwp in self._nwp_info:
            input_feature, output_feature = WsWpFeature.stacking_feature_layer1([nwp], feature_dict, single_source=True)
            x_nwp_df = self.data_preprocessing(x_df[input_feature])
            x_nwp_df, feature_names = XgbWsForecast.add_shift_features(x_nwp_df, nwp + ".ws")
            revise_ws_array = self._xgb_predict(x_nwp_df, self._estimator_[output_feature])
            # add a new line
            result[nwp + ".ws_predict"] = list(revise_ws_array)
            new_feature.append(pd.Series(revise_ws_array, index=x_df.index, name=output_feature))

        # fill nan
        for nwp in self._nwp_info:
            result[nwp + ".ws_predict"] = result[nwp + ".ws_predict"].fillna('999')
            # return nan value index
            row_num = result[(result[nwp + ".ws_predict"] == '999')].index.tolist()
            # fill nan with original value
            for i in range(len(row_num)):
                result.ix[row_num[i], nwp + ".ws_predict"] = x_df.ix[row_num[i]][nwp + ".ws"]
            #result[row_num][nwp + ".ws_predict"] = x_df[row_num][nwp +".ws"]

        name = []
        for nwp in self._nwp_info:
            name.append(nwp + ".ws_predict")

        # non horizon
        result['X_basic.horizon'] = x_df['X_basic.horizon']
        result['X_basic.time'] = x_df['X_basic.time']
        result = pd.concat([result, y_df['Y.ws_tb']], axis=1)
        result = result[(result['X_basic.horizon'] >= 16) & (result['X_basic.horizon'] <= 39)]
        prediction = self._linear_predict(result, name, self._estimator_['combine.ws'])
        prediction_result = pd.DataFrame({'X_basic.horizon': result['X_basic.time'], 'Y.ws_tb': result['Y.ws_tb'],
                                          'prediction': prediction})

        # # add new horizon
        # result['X_basic.horizon'] = x_df['X_basic.horizon']
        # result['X_basic.time'] = x_df['X_basic.time']
        # result = pd.concat([result, y_df['Y.ws_tb']], axis=1)
        # result = result[(result['X_basic.horizon'] >= 16) & (result['X_basic.horizon'] <= 39)]
        # horizon_list = list(range(16, 40))
        # prediction_list = []
        # true_list = []
        # time_list = []
        # for horizon in horizon_list:
        #     prediction = self._linear_predict_horizon(result, name, self._estimator_['combine.ws'], horizon)
        #     true_list.append(result[result['X_basic.horizon'] == horizon]['Y.ws_tb'])
        #     time_list.append(result[result['X_basic.horizon'] == horizon]['X_basic.time'])
        #     prediction_list.append(prediction)
        # prediction_list = np.array(prediction_list).reshape(-1).tolist()
        # true_list = np.array(true_list).reshape(-1).tolist()
        # time_list = np.array(time_list).reshape(-1).tolist()
        # prediction_result = pd.DataFrame({'X_basic.horizon': time_list, 'Y.ws_tb': true_list, 'prediction': prediction_list})

        return prediction_result