from common_misc import load_data_from_pkl
import pandas as pd
import numpy as np
import copy as copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import VarianceThreshold


x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')

# concat by column
data_train = pd.concat([x_train, y_train], axis=1)
data_test = pd.concat([x_test, y_test], axis=1)
# drop out nan value
data_train = data_train.dropna(subset=['Y.ws_tb'])
data_train = data_train[np.isnan(data_train['GFS0.ws']) == False]
data_train = data_train[np.isnan(data_train['WRF0.ws']) == False]

def delete_outlier_ws(data):
    a = np.array(data)
    nums = []
    for i in range(len(a)):
        mean_ws = (a[i][6]+a[i][13]+a[i][20])/3
        if abs(mean_ws-a[i][27]) > 3:
            nums.append(i)
    a = np.delete(a, nums, axis=0)
    data = pd.DataFrame(a)
    return data

def localoutlier_and_dbscan(data):
    a = np.array(data)
    X = a[:,[6,27]]
    y_scaled = preprocessing.scale(X)

    # print(Y_scaled)
    # default eps=0.5, min_samples=5
    # clf=DBSCAN(eps=0.2, metric='euclidean', algorithm='auto', min_samples=10)
    # default n_neighbors=20, contamination=0.1
    clf = LocalOutlierFactor(n_neighbors=200, contamination=0.08)

    y_pred = clf.fit_predict(y_scaled)
    print(clf)
    print(y_pred)

    x = [n[0] for n in X]
    y = [n[1] for n in X]
    # visualization
    plt.scatter(x, y, c=y_pred, marker='*')
    plt.title("outlier detection")
    plt.xlabel("wind speed")
    plt.ylabel("Y.ws_tb")
    plt.legend(["data"])
    plt.show()

