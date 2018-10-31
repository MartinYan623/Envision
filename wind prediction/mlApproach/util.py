#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongjian.liu
@date:   2017/11/15
"""
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
import os

logger = logging.getLogger(__name__)


def train_test_split(x_df, y_df, test_ratio=0.2):
    """将数据集划分为训练集及测试集，划分时保证同一时次发布的数据同在训练集或测试集中"""
    i_set = x_df['i.set']
    is_test = i_set > i_set.iloc[int(round((1 - test_ratio) * len(i_set)))]
    x_test = x_df.loc[is_test]
    y_test = y_df.loc[is_test]
    x_train = x_df.loc[~is_test]
    y_train = y_df.loc[~is_test]
    logger.debug('Splitting data set into training/test set...')
    logger.debug('Totally {} samples, training/test: {}/{}.'.format(len(x_df), len(x_train), len(x_test)))
    return x_train, y_train, x_test, y_test


def load_pkl(master_id, prefix=''):
    """自pkl文件中读取特征数据"""
    logger.debug('Loading feature table from pickle file, master ID: {}'.format(master_id))
    file_path = prefix + os.sep + 'feature_table_' + master_id + '.pkl'
    try:
        feature_table = pd.read_pickle(file_path)
    except IOError:
        logger.error(master_id + ' pickle file doesn''t exist!', exc_info=True)
        raise
    y_label = [x for x in feature_table.columns if x.lower().startswith('y')]
    if not bool(y_label):
        logger.warning('Y data missed.')
        x_df = feature_table
        y_df = None
    else:
        y_df = feature_table[y_label]
        x_df = feature_table.drop(y_label, axis=1)
    return x_df, y_df


def get_nwp_list(column_list):
    """获得所有NWP名称"""
    nwp = []
    for column_name in column_list:
        if column_name.endswith('dist'):
            nwp.append(column_name.split('.')[0])
    return nwp


def feature_clean(x_df, y_df=None, cleaning_x=True):
    """清除含有空值的样本。可选择是否保留X中的空值。"""
    if cleaning_x:
        if isinstance(x_df, pd.DataFrame):
            is_valid = x_df.notnull().all(axis=1)
        else:
            is_valid = x_df.notnull()
    else:
        is_valid = pd.Series(np.ones(len(x_df), dtype=bool), index=x_df.index)
    if y_df is not None:
        if isinstance(y_df, pd.DataFrame):
            is_valid = is_valid & y_df.notnull().all(axis=1)
        else:
            is_valid = is_valid & y_df.notnull()
    x_clean = x_df.loc[is_valid]
    if y_df is not None:
        y_clean = y_df.loc[is_valid]
        return x_clean, y_clean, is_valid
    else:
        return x_clean, is_valid


def create_cv_index(i_set, k_folds, shuffle=False, seed=123):
    """返回k折交叉验证集的标签，划分时保证同一时次发布的数据同在训练集或cv集中"""
    logger.debug('Creating {} cv index into {} folds.'.format('shuffle' if shuffle else 'sequential', k_folds))
    i_set_unique = np.unique(i_set)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(i_set_unique)
    cv_index = np.ones_like(i_set, dtype=int) * -1
    for k in range(k_folds):
        for m in i_set_unique[k::k_folds]:
            cv_index[i_set == m] = k
    return cv_index


def create_cv_generator(i_set, k_folds, shuffle=False, seed=None):
    """返回k折交叉验证集的生成器，划分时保证同一时次发布的数据同在训练集或cv集中"""
    cv_index = create_cv_index(i_set, k_folds, shuffle=shuffle, seed=seed)
    for m in range(k_folds):
        is_cv = cv_index == m
        yield np.where(~is_cv)[0], np.where(is_cv)[0]


def add_feature_wdcut(x_df, n_sector, one_hot_encoding=True, concat=True):
    """风向离散化，可输入扇区数"""
    logger.debug('Splitting data into {} wind direction sectors...'.format(n_sector))
    logger.debug('Adding new feature: {nwp}.wd_cut')
    bins = np.arange(0, 360 + 1e-5, 360 / n_sector)
    bin_labels = ['s{}'.format(n) for n in range(n_sector)]
    nwp_list = get_nwp_list(x_df.columns)
    tmp_dict = {}
    for nwp in nwp_list:
        tmp_dict[nwp + '.wd_cut'] = pd.cut(x_df[nwp + '.wd'], bins, labels=bin_labels)
    df_wdcut = pd.DataFrame(tmp_dict)
    if one_hot_encoding:
        df_wdcut = pd.get_dummies(df_wdcut)
    if concat:
        return pd.concat([x_df, df_wdcut], axis=1)
    else:
        return df_wdcut


def add_feature_one_hot_horizon(x_df, concat=True):
    """对预报步长做one_hot_encoding"""
    logger.debug('One hot encoding for X_basic.horizon')
    one_hot_df = pd.get_dummies(x_df['X_basic.horizon'], prefix='X_basic.horizon')
    if concat:
        return pd.concat([x_df, one_hot_df], axis=1)
    else:
        return one_hot_df


def add_feature_rho_crossed(x_df, concat=True):
    """添加密度与风速交叉项"""
    logger.debug('Adding density-speed crossed features...')
    nwp_list = get_nwp_list(x_df.columns)
    tmp_dict = {}
    for nwp in nwp_list:
        tmp_dict[nwp + '.rho_ws'] = x_df[nwp + '.ws'] * x_df[nwp + '.rho']
        tmp_dict[nwp + '.rho_ws2'] = x_df[nwp + '.ws'] ** 2 * x_df[nwp + '.rho']
        tmp_dict[nwp + '.rho_ws3'] = x_df[nwp + '.ws'] ** 3 * x_df[nwp + '.rho']
    tmp_df = pd.DataFrame(tmp_dict)
    if concat:
        return pd.concat([x_df, tmp_df], axis=1)
    else:
        return tmp_df


def add_feature_shift(df, param, shift, concat=True):
    """指定要素沿时间轴平移"""
    logger.debug('Shifting %s by %d interval...' % (param, shift))
    if shift > 0:
        new_param = param + '_p{}'.format(shift)
    else:
        new_param = param + '_n{}'.format(abs(shift))
    new_df = df[['i.set', 'X_basic.time', param]].copy()
    new_df.columns = ['i.set', 'X_basic.time', new_param]
    new_df['X_basic.time'] = new_df['X_basic.time'] + timedelta(hours=shift)
    merged_df = pd.merge(df, new_df, how='left', on=['i.set', 'X_basic.time'])
    merged_df.index = df.index
    is_nan = merged_df[new_param].isnull()
    merged_df.loc[is_nan, new_param] = merged_df.loc[is_nan, param]
    if concat:
        return merged_df
    else:
        return merged_df[new_param]