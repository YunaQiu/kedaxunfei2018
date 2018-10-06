#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
from datetime import *
import json, random, os

from sklearn.preprocessing import *
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

from utils import *
from feature1 import feaFactory


class LgbModel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
        	'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.01,
        	'num_leaves': 25,
            'max_depth': -1,
            # 'min_data_in_leaf': 120,
            # 'feature_fraction': 0.9,
            # 'bagging_fraction': 0.9,
        	# 'bagging_freq': 5,
            'verbose': 0,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea

    def train(self, X, y, num_round=8000, validX=None, validy=None, early_stopping=10, verbose=True, params={}):
        trainData = lgb.Dataset(X, label=y, categorical_feature=self.cateFea)
        trainParam = self.params
        trainParam.update(params)
        if isinstance(validX, pd.DataFrame):
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=validData, early_stopping_rounds=early_stopping, verbose_eval=verbose)
        else:
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, verbose_eval=verbose)
        self.bst = bst
        return bst.best_iteration

    def cv(self, X, y, nfold=5, num_round=8000, early_stopping=10, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, categorical_feature=self.cateFea)
        result = lgb.cv(trainParam, trainData, num_boost_round=num_round, nfold=nfold, early_stopping_rounds=early_stopping, verbose_eval=verbose)
        return result

    def predict(self, X):
        return self.bst.predict(X)

    def feaScore(self, show=True):
        scoreDf = pd.DataFrame({'fea': self.feaName, 'importance': self.bst.feature_importance()})
        scoreDf.sort_values(['importance'], ascending=False, inplace=True)
        if show:
            print(scoreDf)
        return scoreDf

    def gridSearch(self, X, y, validX, validy, nFold=5, verbose=0):
        paramsGrids = {
            'num_leaves': [5*i for i in range(2,9)],
            # 'max_depth': list(range(3,8)),
            'min_data_in_leaf': [20*i for i in range(1,10)],
            'bagging_fraction': [1-0.05*i for i in range(0,5)],
            'bagging_freq': list(range(0,11,2)),
        }
        def getEval(params):
            iter = self.train(X, y, validX=validX, validy=validy, params=params, verbose=verbose)
            return metrics.log_loss(validy, self.predict(validX)), iter
        for k,v in paramsGrids.items():
            resultDf = pd.DataFrame({k: v})
            resultDf['metric_mean'] = list(map(lambda x: getEval({k: x}), v))
            print(resultDf)
        exit()

def main():
    # 获取特征工程数据集
    if not os.path.isfile("../temp/fea1.csv"):
        df = importDf("../data/round1_iflyad_train.txt")
        df['flag'] = 0
        predictDf = importDf("../data/round1_iflyad_test_feature.txt")
        predictDf['flag'] = -1
        originDf = pd.concat([df,predictDf], ignore_index=True)
        originDf = feaFactory(originDf)
        exportResult(originDf, "../temp/fea1.csv")
    else:
        originDf = pd.read_csv("../temp/fea1.csv")
    print("get feature dataset: finished!")

    # 构建训练及测试数据集
    originDf = labelEncoding(originDf, ['advert_id','creative_id','creative_tp_dnf','advert_industry_inner','inner_slot_id','creative_area','slot_prefix','city','province'])
    fea = [
        'advert_id','advert_industry_inner','creative_has_deeplink','creative_id','creative_is_jump','creative_tp_dnf','creative_type',
        'app_cate_id','inner_slot_id',
        'city','province','carrier','devtype','nnt','os',

        'day','hour','date','online_days',
        'advert_industry_inner1','creative_area','advert_showed','creative_area_today_num','creative_area_his_ctr','creative_today_num','creative_his_ctr',
        'slot_prefix','slot_today_num','slot_his_ctr',
        'is_wifi','nnt_gtype','osv1','ios_osv1','android_osv1',
        'tags21_len','tags30_len','tagsLen10_len','tagsAg_len','tagsGd_len','tagsMz_len',
        ]
    cateFea = [
        'advert_id','advert_industry_inner','creative_id','creative_tp_dnf','creative_type',
        'app_cate_id','inner_slot_id',
        'city','province','carrier','devtype','nnt','os',

        'advert_industry_inner1','creative_area',
        'slot_prefix',
        ]
    df = originDf[originDf.flag>=0]
    trainDf = df[df.day < 6]
    validDf = df[df.day == 6]
    predictDf = originDf[originDf.flag==-1]

    # 训练模型
    model = LgbModel(fea, cateFea=cateFea)
    # model.gridSearch(trainDf[fea], trainDf['click'], validDf[fea], validDf['click'])
    iterNum = model.train(trainDf[fea], trainDf['click'], validX=validDf[fea], validy=validDf['click'])
    model.train(df[fea], df['click'], num_round=iterNum)
    model.feaScore()

    # 预测结果
    predictDf['predicted_score'] = model.predict(predictDf[fea])
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    exportResult(predictDf[['instance_id','predicted_score']], "../result/lgb1.csv")

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
