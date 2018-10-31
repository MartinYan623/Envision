#!/usr/bin/env python
# encoding: utf-8
"""
revsied based on mlApproach.xgb_util.py
"""
import xgboost as xgb
import numpy as np
import pandas as pd
import itertools


def hyperparameter_search(dtrain, dtest, grid_params, seed=None, num_boost_round=300,
                          early_stopping_rounds=8, verbose_eval=False, **kwargs):
    """
    :param xgb.DMatrix dtrain: 训练集
    :param xgb.DMatrix dtest: 测试集
    :param dict or list grid_params: 参数搜索空间
    :param int n_iter: 模型迭代次数
    :param int seed: 随机数种子
    :param int num_boost_round:
    :param int early_stopping_rounds: 早停轮次
    :param bool verbose_eval: xgb迭代输出静默
    :param kwargs: 其他参数
    检索xgb参数空间，返回score列表，最优树数目列表，对应参数列表，特征重要性列表
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(1914)
    param_list, score_list, best_ntree_list, importance_by_gain = [], [], [], []
    for param in build_parameter_combination(grid_params):
        bst = xgb.train(param, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dtest, 'eval')],
                        verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds, **kwargs)
        param_list.append(param)
        score_list.append(bst.best_score)
        best_ntree_list.append(bst.best_ntree_limit)
        importance_by_gain.append(bst.get_score(importance_type='gain'))
    return score_list, best_ntree_list, param_list, importance_by_gain


def score_stats_by_params(score_list, params_list):
    arr_score = pd.Series(score_list)
    df_params = pd.DataFrame(params_list)
    result = []
    for param in df_params.columns:
        result.append(arr_score.groupby(df_params[param]).agg(['mean', 'std']))
    return pd.concat(result, keys=df_params.columns)


def cross_prediction(params, x_df, y_df, idx_fold, num_boost_round, missing=None, return_bst=False):
    prediction = np.zeros_like(idx_fold, dtype=float) * np.nan
    for idx in np.unique(idx_fold):
        dtrain = xgb.DMatrix(x_df.loc[idx_fold != idx].values, label=y_df.loc[idx_fold != idx].values, missing=missing)
        dtest = xgb.DMatrix(x_df.loc[idx_fold == idx].values, missing=missing)
        bst = xgb.train(params, dtrain, num_boost_round)
        prediction[idx_fold == idx] = bst.predict(dtest)
    if return_bst:
        dtrain = xgb.DMatrix(x_df, label=y_df.values, missing=missing)
        bst = xgb.train(params, dtrain, num_boost_round)
        return prediction, bst
    else:
        return prediction


def build_parameter_combination(grid_params):
    keys = grid_params.keys()
    param_list = [grid_params[key] for key in keys]
    all_value_combination = list(itertools.product(*param_list))
    all_param_list = []
    for value_list in all_value_combination:
        all_param_list.append(dict(zip(keys, value_list)))
    return all_param_list
