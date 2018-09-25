from common_misc import load_data_from_pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
# load data
x_df, y_df=load_data_from_pkl('data/turbine_314e3ca4bd2345c1bc4f649f313d0b18.pkl')
#print(x_df)
#print(y_df)

# concat by column
data = pd.concat([x_df,y_df],axis=1)
#print(data)


ws_map = [
    {'lower': 0, 'upper': 4, 'val': 0},
    {'lower': 4, 'upper': 12, 'val': 1},
    {'lower': 12, 'upper': 99, 'val': 2},
]

# attribute reduction
def numerical_to_bin(data, attr, val_map):
    result = data.copy(deep=True)
    for the_map in val_map:
        lower = the_map['lower']
        upper = the_map['upper']
        val = the_map['val']
        result.loc[np.logical_and(
            data[attr] >= lower, data[attr] < upper), attr] = val
    return result

EC0_predictors=['EC0.ws','EC0.wd','EC0.tmp','EC0.pres','EC0.rho']
GFS0_predictors=['GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.pres','GFS0.rho']
WRF0_predictors=['WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.pres','WRF0.rho']
# prediction model
def prediction(train,test,predictors):
    #x_train, x_test, y_train, y_test = train_test_split(data[predictors], data['Y.ws_tb'], random_state=1,test_size=.2)
    x_train=train[predictors]
    y_train=train['Y.ws_tb']
    x_test= test[predictors]
    y_test= test['Y.ws_tb']
    clf = OneVsRestClassifier(RandomForestClassifier(min_samples_leaf=3, n_estimators=50, min_samples_split=10, max_depth=10))
    clf= OneVsRestClassifier(GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3))
    clf.fit(x_train, y_train)
    print('The accuracy of training set:', accuracy_score(y_train, clf.predict(x_train)))
    print('The accuracy of testing set:', accuracy_score(y_test, clf.predict(x_test)))


def EC0_prediction(data,start_day=1,end_day=394,probability=0.8,split=4):
    train = pd.DataFrame(columns=['EC0.ws','EC0.wd','EC0.tmp','EC0.pres','EC0.rho','Y.ws_tb'])
    test = pd.DataFrame(columns=['EC0.ws','EC0.wd','EC0.tmp','EC0.pres','EC0.rho','Y.ws_tb'])
    count=0
    for i in range(start_day,end_day,1):
        data_1=data[data['i.set']==i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum()/289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            count=count+1
            if count%split==0:
                test=pd.concat([test,data_1],names=['EC0.ws','EC0.wd','EC0.tmp','EC0.pres','EC0.rho','Y.ws_tb'])
            else:
                train=pd.concat([train, data_1],names=['EC0.ws','EC0.wd','EC0.tmp','EC0.pres','EC0.rho','Y.ws_tb'])
    prediction(train,test,EC0_predictors)

def GFS0_prediction(data, start_day=1,end_day=394,probability=0.8,split=4):
    data = data[np.isnan(data['GFS0.ws']) == False]
    train = pd.DataFrame(columns=['GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.pres','GFS0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.pres','GFS0.rho', 'Y.ws_tb'])
    count = 0
    for i in range(start_day,end_day,1):
        data_1=data[data['i.set']==i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum()/289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            count=count+1
            if count%split==0:
                test=pd.concat([test,data_1],names=['GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.pres','GFS0.rho','Y.ws_tb'])
            else:
                train=pd.concat([train, data_1],names=['GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.pres','GFS0.rho','Y.ws_tb'])
    prediction(train,test,GFS0_predictors)

def WRF0_prediction(data,start_day=1,end_day=394,probability=0.8,split=4):
    data = data[np.isnan(data['WRF0.ws']) == False]
    train = pd.DataFrame(columns=['WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.pres','WRF0.rho', 'Y.ws_tb'])
    test = pd.DataFrame(columns=['WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.pres','WRF0.rho', 'Y.ws_tb'])
    count = 0
    for i in range(start_day, end_day, 1):
        data_1 = data[data['i.set'] == i]
        if (~np.isnan(data_1['Y.ws_tb'])).sum() / 289 > probability:
            # drop the data whose 'Y.ws_tb' is nan
            data_1 = data_1.dropna(subset=['Y.ws_tb'])
            count = count + 1
            if count % split == 0:
                test = pd.concat([test, data_1],names=['WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.pres','WRF0.rho', 'Y.ws_tb'])
            else:
                train = pd.concat([train, data_1],names=['WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.pres','WRF0.rho', 'Y.ws_tb'])
    prediction(train, test, WRF0_predictors)

data = numerical_to_bin(data, 'Y.ws_tb', ws_map)

# check data if there is nan value
#nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:35])
#nulls.columns = ['Null Count']
#nulls.index.name = 'Feature'
#print(nulls)

#EC0_prediction(data)
#GFS0_prediction(data)
WRF0_prediction(data)


"""
    # 用到的特征
    # 1.线性回归
    # alg = LinearRegression()
    # 2.逻辑回归
    # alg= LogisticRegression(random_state=1)
    # 3.梯度提升树 迭代决策树
    # alg= GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
    # 4.弱学习算法”提升(boost)为“强学习算法
    # alg= AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=3, class_weight=cw), n_estimators=50)
    # 5.神经网络(多层感知器分类器)
    # alg = MLPClassifier(hidden_layer_sizes=(100, 50, 50), solver='sgd', max_iter=400, learning_rate="adaptive", alpha=0.5, early_stopping=True)
    # 6.随机森林
    # alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
    # 7.朴素贝叶斯(MultinomialNB 假设特征的先验概率为多项式分布 另外还有GaussianNB 假设特征的先验概率为正态分布；MultinomialNB 假设特征的先验概率为多项式分布)
    # alg = MultinomialNB(alpha=0.01)
    # 8.KNN
    # alg = KNeighborsClassifier(n_neighbors=10)
    # 9.决策树
    # alg = tree.DecisionTreeClassifier()
    # 10.支持向量机SVM
    # alg = SVC(kernel='rbf', probability=True)
"""
