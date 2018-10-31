#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongjian.liu
@date:   2017/11/16
"""
import xgboost as xgb
from functools import reduce
import numpy as np
import pandas as pd
import operator
import time


def hyperparameter_search(dtrain, dtest, grid_params, n_iter=32, seed=None, num_boost_round=300,
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
    param_list, score_list, best_ntree_list, importance_by_gain = [], [], [], []
    for param in _ParameterSampler(grid_params, n_iter):
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
        dtrain = xgb.DMatrix(x_df.values, label=y_df.values, missing=missing)
        bst = xgb.train(params, dtrain, num_boost_round)
        return prediction, bst
    else:
        return prediction


class _ParameterSampler(object):
    def __init__(self, param_distribution, n_iter):
        self.param_distr = param_distribution
        self.n_iter = n_iter
        self.p = None
        if isinstance(self.param_distr, list):
            n_sets = []
            for param_dict in self.param_distr:
                n_sets.append(reduce(operator.mul, [len(x) for x in param_dict.values()]))
            self.p = np.array(n_sets) / sum(n_sets)

    def __iter__(self):
        for kk in range(self.n_iter):
            if isinstance(self.param_distr, list):
                i_grid = np.random.choice(range(len(self.param_distr)), p=self.p)
                yield self.random_sample(self.param_distr[i_grid])
            else:
                yield self.random_sample(self.param_distr)

    @staticmethod
    def random_sample(params_grid):
        result = {}
        for k, v in params_grid.items():
            result[k] = np.random.choice(v)
        return result
