# coding=utf-8
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from power_forecast_common.xgb_model import XgbForecast
from power_forecast_common.wswp_feature import WsWpFeature
from power_forecast_common.evaluation_misc import wind_std, wind_std_distribution
from xgb_ws_forecast import XgbWsForecast

logger = logging.getLogger(__name__)


class XgbSVRWsForecast(XgbWsForecast):

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
        lr = SVR(C=1.0, epsilon=0.2)
        combine = lr.fit(new_data[name], new_data['Y.ws_tb'])
        self._estimator_['combine.ws'] = combine
        new_data['combine.ws'] = lr.predict(new_data[name])
        cur_std = wind_std(new_data['Y.ws_tb'], new_data['combine.ws'])
        print('the std on training date after adding linear layer is: ' + str(cur_std))

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
                result.ix[row_num[i], nwp + ".ws_predict"] = x_df.ix[row_num[i]][nwp +".ws"]

        name = []
        for nwp in self._nwp_info:
            name.append(nwp + ".ws_predict")

        prediction, score = self._linear_predict(result, y_df['Y.ws_tb'], name, self._estimator_['combine.ws'])
        print(prediction)

        print('evaluate model r-squared value is: ' + str(score))
        cur_std = wind_std(y_df['Y.ws_tb'], prediction)
        print('the std on testing data after adding linear layer is:' + str(cur_std))

        return cur_std