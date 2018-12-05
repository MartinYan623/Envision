# coding=utf-8
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class WsWpFeature:
    def __init__(self, train_frequency=10, delta_hour=3, nwp_list=[]):
        self._train_frequency_ = train_frequency
        self._nwp_list_ = nwp_list
        self._feature_dict_ = {}
        self._delta_hour_ = delta_hour

    def transform(self, x_df):
        x_df, time_feature_names = self._time_feature(x_df)
        self.add_feature("common", time_feature_names)
        self._get_nwp_basic(x_df.columns)
        x_df = self._delta_features(x_df)
        x_df = self._direction_feature(x_df)
        #x_df = self._direction_feature2(x_df)
        return x_df, self._feature_dict_

    def _get_nwp_basic(self, data_columns):
        for nwp in self._nwp_list_:
            nwp_entries = [col for col in data_columns if nwp in col and "shift" not in col]
            if "nwp_time" in nwp_entries:
                nwp_entries.remove("nwp_time")
            self.add_feature(nwp, nwp_entries)

    def convert_time_feature(self, x_df, time_type="month"):
        if time_type == "month":
            value_array = x_df["X_basic.time"].dt.month
            max_value = 12
        elif time_type == "day":
            value_array = x_df["X_basic.time"].dt.day
            max_value = 30
        elif time_type == "hour":
            value_array = x_df["X_basic.time"].dt.hour * 60 + x_df["X_basic.time"].dt.minute
            max_value = 24 * 60

        feature_names = []
        feature_list = []
        pi_array = [WsWpFeature.map_to_degree(value, max_value) for value in value_array]
        cur_name = "{}_sin".format(time_type)
        feature_list.append(pd.Series(np.sin(pi_array), index=x_df.index, name=cur_name))
        feature_names.append(cur_name)

        cur_name = "{}_cos".format(time_type)
        feature_list.append(pd.Series(np.cos(pi_array), index=x_df.index, name=cur_name))
        feature_names.append(cur_name)
        return feature_list, feature_names

    def _time_feature(self, x_df):
        # time: hour of the day, day of the month, month of the year
        time_features = []
        feature_names = []

        time_type_list = ["month", "day", "hour"]
        for time_type in time_type_list:
            cur_feature_list, cur_feature_names = self.convert_time_feature(x_df, time_type=time_type)
            time_features.extend(cur_feature_list)
            feature_names.extend(cur_feature_names)
        x_df = pd.concat([x_df, *time_features], axis=1)
        return x_df, feature_names

    def add_feature(self, key, values):
        if key in self._feature_dict_:
            self._feature_dict_[key].extend(values)
        else:
            self._feature_dict_.update({key: values})

    def remove_feature(self, key, value):
        assert key in self._feature_dict_
        if value in self._feature_dict_[key]:
            self._feature_dict_[key].remove(value)

    def _direction_feature(self, x_df):
        direction_feature = []
        for nwp in self._nwp_list_:
            feature_names = []
            pi_array = [WsWpFeature.map_to_degree(value, 360) for value in x_df[nwp + '.wd']]
            sin_array = np.sin(pi_array)
            cur_name = "{}.wd_sin".format(nwp)
            direction_feature.append(pd.Series(sin_array, index=x_df.index, name=cur_name))
            feature_names.append(cur_name)

            cos_array = np.cos(pi_array)
            cur_name = "{}.wd_cos".format(nwp)
            direction_feature.append(pd.Series(cos_array, index=x_df.index, name=cur_name))
            feature_names.append(cur_name)
            self.add_feature(nwp, feature_names)
            self.remove_feature(nwp, nwp + ".wd")
        x_df = pd.concat([x_df, *direction_feature], axis=1)
        return x_df

    def _direction_feature2(self, x_df):
        basic_item = ["wd"]
        for nwp in self._nwp_list_:
            if "IBM" in nwp:
                continue
            if "WRF" in nwp:
                delta_hour = 1
            else:
                delta_hour = 3
            for item in basic_item:
                cur_columns = [nwp + "." + item]
                cur_columns.extend(self.get_nwp_nearby_column(x_df, nwp, item))
                cur_shift_features, shift_names = self.shift_features(x_df, cur_columns, [delta_hour], False)
                cur_delta_features = cur_shift_features[shift_names[1:]].values - \
                                     np.tile(cur_shift_features[shift_names[0]].values, (len(shift_names) - 1, 1)).T
                cur_delta_names = [name + "_dD1" for name in shift_names[1:]]
                delta_features = pd.DataFrame(cur_delta_features, index=x_df.index, columns=cur_delta_names)
                x_df = pd.concat([x_df, delta_features], axis=1)

            feature_names = []
            for item in cur_delta_names:
                pi_array = [WsWpFeature.map_to_degree(value, 360) for value in x_df[item]]
                sin_array = np.sin(pi_array)
                cur_name = "{}._sin".format(item)
                direction_feature_sin = pd.Series(sin_array, index=x_df.index, name=cur_name)
                feature_names.append(cur_name)
                x_df = pd.concat([x_df, direction_feature_sin], axis=1)

                cos_array = np.cos(pi_array)
                cur_name = "{}._cos".format(item)
                direction_feature_cos = pd.Series(cos_array, index=x_df.index, name=cur_name)
                feature_names.append(cur_name)
                x_df = pd.concat([x_df, direction_feature_cos], axis=1)

            self.add_feature(nwp, feature_names)
        return x_df


    @staticmethod
    def shift_features(x_df, shift_cols, shift_number_list, concat=True):
        shift_features = []
        feature_names = []
        for col in shift_cols:
            data_array = x_df[col]
            for num in shift_number_list:
                cur_array = data_array.shift(num)
                cur_name = "{}_dH{}".format(col, num)
                shift_features.append(pd.Series(cur_array, index=x_df.index, name=cur_name))
                feature_names.append(cur_name)
        if concat:
            x_df = pd.concat([x_df, *shift_features], axis=1)
            return x_df, feature_names
        else:
            return pd.DataFrame(shift_features).T, feature_names

    def get_nwp_nearby_column(self, x_df, nwp, item):
        nearby_colums = []
        for col in x_df.columns:
            if nwp in col and "shift" in col and item in col:
                nearby_colums.append(col)
        return nearby_colums

    def _delta_features(self, x_df):
        basic_item = ["pres"]
        delta_features = []
        for nwp in self._nwp_list_:
            if "IBM" in nwp:
                continue
            if "WRF" in nwp:
                delta_hour = 1
            else:
                delta_hour = 3

            for item in basic_item:
                cur_columns = [nwp + "." + item]
                cur_columns.extend(self.get_nwp_nearby_column(x_df, nwp, item))
                cur_shift_features, shift_names = self.shift_features(x_df, cur_columns, [delta_hour], False)
                cur_delta_features = cur_shift_features[shift_names[1:]].values - \
                                     np.tile(cur_shift_features[shift_names[0]].values, (len(shift_names)-1, 1)).T
                cur_delta_names = [name + "_dD1" for name in shift_names[1:]]
                delta_features.append(pd.DataFrame(cur_delta_features, index=x_df.index, columns=cur_delta_names))
                self.add_feature(nwp, cur_delta_names)
        x_df = pd.concat([x_df, *delta_features], axis=1)
        return x_df

    # def _delta_features(self, x_df):
    #     basic_columns = ["ws", "wd", "tmp", "pres", "rho"]
    #     delta_step = self._delta_hour_ * int(60.0 / self._train_frequency_)
    #     delta_features = []
    #     for nwp in self._nwp_list_:
    #         for basic in basic_columns:
    #             col_name = nwp + "." + basic
    #             delta_array = WsWpFeature.first_order_derivative(x_df[col_name], delta_step)
    #             cur_name = "{}.{}_1st_order_{}".format(nwp, basic, delta_step)
    #             delta_features.append(pd.Series(delta_array, index=x_df.index, name=cur_name))
    #             self.add_feature(nwp, [cur_name])
    #     x_df = pd.concat([x_df, *delta_features], axis=1)
    #     return x_df

    @staticmethod
    def speed_rho_individual_feature(nwp_list, x_df, concat=True):
        tmp_dict = {}
        feature_names = []
        for nwp in nwp_list:
            cols = [nwp + '.revised_ws', nwp + '.rho']
            is_valid = x_df[cols].notnull().all(axis=1)
            array = x_df.loc[is_valid, nwp + '.revised_ws'] * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array

            cur_name = nwp + '.rho_revised_ws'
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)

            array = x_df.loc[is_valid, nwp + '.revised_ws'] ** 2 * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array
            cur_name = nwp + '.rho_revised_ws2'
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)

            array = x_df.loc[is_valid, nwp + '.revised_ws'] ** 3 * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array
            cur_name = nwp + '.rho_revised_ws3'
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)
        tmp_df = pd.DataFrame(tmp_dict)
        if concat:
            return pd.concat([x_df, tmp_df], axis=1), feature_names
        else:
            return tmp_df, feature_names

    @staticmethod
    def speed_rho_group_feature(x_df, speed_col, nwp_list, concat=True):
        tmp_dict = {}
        cols = [nwp + '.rho' for nwp in nwp_list] + [speed_col]
        is_valid = x_df[cols].notnull().all(axis=1)
        feature_names = []
        for nwp in nwp_list:
            array = x_df.loc[is_valid, speed_col] * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array
            cur_name = '{}.rho_{}1'.format(nwp, speed_col)
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)

            array = x_df.loc[is_valid, speed_col] ** 2 * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array
            cur_name = '{}.rho_{}2'.format(nwp, speed_col)
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)

            array = x_df.loc[is_valid, speed_col] ** 3 * x_df.loc[is_valid, nwp + '.rho']
            add_feature = np.nan * np.zeros(len(x_df))
            add_feature[is_valid] = array
            cur_name = '{}.rho_{}3'.format(nwp, speed_col)
            tmp_dict[cur_name] = add_feature
            feature_names.append(cur_name)

        tmp_df = pd.DataFrame(tmp_dict)
        if concat:
            return pd.concat([x_df, tmp_df], axis=1), feature_names
        else:
            return tmp_df, feature_names

    @staticmethod
    def first_order_derivative(data_array, delta_step):
        shifted_array = data_array.shift(delta_step)
        delta_array = shifted_array - data_array
        return delta_array

    @staticmethod
    def map_to_degree(degree, max_value):
        return degree * 2 * np.pi / max_value

    # @staticmethod
    # def stacking_feature_layer1(nwp_list, feature_dict, single_source=False):
    #     keys = ["ws", "wd_sin", "wd_cos", "tmp", "pres", "rho", "dist"]
    #     input_feature = []
    #     for nwp in nwp_list:
    #         if "IBM" in nwp:
    #             # IBM only has wind data
    #             cur_features = [nwp + "." + item for item in ["ws", "wd_sin", "wd_cos"]]
    #         else:
    #             cur_features = [nwp + "." + item for item in keys]
    #         input_feature.extend(cur_features)
    #
    #     common_feature = [feature for feature in feature_dict["common"] if "month" not in feature]
    #     input_feature.extend(common_feature)
    #     if single_source:
    #         output_feature = "{}.revised_ws".format(nwp_list[0])
    #     else:
    #         output_feature = "all.revised_ws"
    #     return input_feature, output_feature

    @staticmethod
    def stacking_feature_layer1(nwp_list, feature_dict, single_source=False):
        input_feature = feature_dict["common"] + feature_dict[nwp_list[0]]
        if nwp_list[0] + ".nwp_time" in input_feature:
            input_feature.remove(nwp_list[0] + ".nwp_time")
        if single_source:
            output_feature = "{}.revised_ws".format(nwp_list[0])
        else:
            output_feature = "all.revised_ws"
        return input_feature, output_feature

    @staticmethod
    def stacking_feature_layer2_turbine(x_df, nwp_list):
        # for turbine-based model
        input_feature = [col for col in x_df.columns if ".revised_ws" in col]
        for nwp in nwp_list:
            input_feature.extend([nwp + '.rho_revised_ws', nwp + ".rho_revised_ws2", nwp + ".rho_revised_ws3"])
        return input_feature, "power"

    @staticmethod
    def stacking_feature_layer2_farm(nwp_list, speed_col):
        # for farm-based model
        input_feature = [speed_col]
        for nwp in nwp_list:
            input_feature.extend(["{}.rho_{}1".format(nwp, speed_col),
                                  "{}.rho_{}2".format(nwp, speed_col),
                                  "{}.rho_{}3".format(nwp, speed_col)])
        return input_feature, "power"