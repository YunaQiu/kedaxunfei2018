#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
from scipy import sparse
import csv
from datetime import *
import json, random, os

from sklearn.preprocessing import *
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectPercentile
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

from utils import *
from feature2 import feaFactory, userTagsMatrix, addTime


class LgbModel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
        	'num_leaves': 150,
            'max_depth': -1,
            'min_data_in_leaf': 80,
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
        	'bagging_freq': 1,
            'verbose': 0,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea

    def train(self, X, y, num_round=8000, validX=None, validy=None, early_stopping=10, verbose=True, params={}):
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        trainParam = self.params
        trainParam.update(params)
        if isinstance(validX, (pd.DataFrame, sparse.csr_matrix)):
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=[trainData,validData], valid_names=['train', 'valid'], early_stopping_rounds=early_stopping, verbose_eval=verbose)
        else:
            bst = lgb.train(trainParam, trainData, valid_sets=trainData, num_boost_round=num_round, verbose_eval=verbose)
        self.bst = bst
        return bst.best_iteration

    def cv(self, X, y, nfold=5, num_round=8000, early_stopping=10, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        result = lgb.cv(trainParam, trainData, feature_name=self.feaName, categorical_feature=self.cateFea, num_boost_round=num_round, nfold=nfold, early_stopping_rounds=early_stopping, verbose_eval=verbose)
        return result

    def predict(self, X):
        return self.bst.predict(X)

    def feaScore(self, show=True):
        scoreDf = pd.DataFrame({'fea': self.feaName, 'importance': self.bst.feature_importance()})
        scoreDf.sort_values(['importance'], ascending=False, inplace=True)
        if show:
            print(scoreDf[scoreDf.importance>0])
        return scoreDf

    def gridSearch(self, X, y, validX, validy, nFold=5, verbose=0):
        paramsGrids = {
            'num_leaves': [10*i for i in range(5,30)],
            # 'max_depth': list(range(3,8)),
            'min_data_in_leaf': [20*i for i in range(1,10)],
            'bagging_fraction': [1-0.05*i for i in range(0,5)],
            'bagging_freq': list(range(0,10)),
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
    FEA_PATH = "../temp/fea2.csv"
    TAGS_PATH = "../temp/user_tags2.npz"
    TAGS_NAME_PATH = "../temp/user_tags_name2.txt"
    if not os.path.isfile(FEA_PATH):
        df1 = importDf("../data/round1_iflyad_train.txt")
        df1['flag'] = 1
        df2 = importDf("../data/round2_iflyad_train.txt")
        df2['flag'] = 2
        df = pd.concat([df1, df2], ignore_index=True)
        df.drop_duplicates(subset=['instance_id'], inplace=True)
        predictDf = importDf("../data/round2_iflyad_test_feature.txt")
        predictDf['flag'] = -1
        originDf = pd.concat([df,predictDf], ignore_index=True)
        originDf = feaFactory(originDf)
        originDf.index = list(range(len(originDf)))
        exportResult(originDf, FEA_PATH)
    else:
        originDf = pd.read_csv(FEA_PATH)
        originDf = addTime(originDf)
    if not os.path.isfile(TAGS_PATH):
        tagsMatrix, tagsName = userTagsMatrix(originDf['user_tags'])
        sparse.save_npz(TAGS_PATH, tagsMatrix)
        fp = open(TAGS_NAME_PATH, "w")
        fp.write(",".join(tagsName))
        fp.close()
    else:
        tagsMatrix = sparse.load_npz(TAGS_PATH)
        fp = open(TAGS_NAME_PATH)
        try:
            tagsName = fp.read().split(",")
        finally:
            fp.close()
    print("load feature dataset: finished!")


    # 筛选及处理特征
    tagSelecter = SelectPercentile(chi2, percentile=20)
    tagSelecter.fit(tagsMatrix[originDf[originDf.flag>=0].index.tolist()], originDf[originDf.flag>=0]['click'])
    tagsMatrix = tagSelecter.transform(tagsMatrix)
    tagsName = np.array(tagsName)[tagSelecter.get_support()]
    cateFea = [
        'adid','advert_id','orderid','advert_industry_inner','campaign_id','creative_id','creative_type','creative_tp_dnf',
        'app_cate_id','f_channel','app_id','inner_slot_id',
        'city','province','carrier','nnt','devtype','os','osv','make','model',

        'advert_industry_inner1','creative_dpi',
        'slot_prefix',
        'region',
        ]
    numFea = [
        'creative_width','creative_height','creative_has_deeplink','creative_is_jump',

        'hour','hour_minute','minute',#'online_days','day',
        'creative_area',#'creative_his_ctr',#'ad_his_ctr','creative_area_his_ctr',# 'advert_showed',#'ad_today_num','creative_area_today_num','creative_today_num',
        'app_tail_number',#'slot_his_ctr',# 'slot_today_num',
        'cityCode','osv1','ios_osv1','android_osv1','isDirectCity','nnt_gtype',#'city_his_ctr',#'city_today_num','is_wifi',
        'tags_num','tags21_len','tags30_len','tagsLen10_len','tagsAg_len','tagsGd_len','tagsMz_len',
        # 'ad_today_num_ratio','dpi_today_num_ratio','creative_today_num_ratio','slot_today_num_ratio','app_today_num_ratio','city_today_num_ratio',
        'ad_num_ratio','dpi_num_ratio','creative_num_ratio','slot_num_ratio','app_num_ratio','city_num_ratio',
    ]
    tagFea = ['tag_'+x for x in tagsName]
    onehotList = ['creative_type','advert_industry_inner1','slot_prefix','region','carrier','nnt','devtype','os']
    originDf = addOneHot(originDf, onehotList)
    onehotFea = []
    # for x in onehotList:
    #     onehotFea.extend(["%s_%s"%(x,v) for v in originDf[x].dropna().unique()])
    # startTime = datetime.now()
    # print('start!!')
    # onehotCsr = pd.get_dummies(originDf[['inner_slot_id']], columns=['inner_slot_id'], sparse=True)
    # print('onehot:', startTime.now() - startTime)
    # selecter = SelectPercentile(percentile=5)
    # selecter.fit(onehotCsr[originDf.flag>=0], originDf[originDf.flag>=0]['click'])
    # print('selecter:', startTime.now() - startTime)
    # slotOnehotFea = onehotCsr.columns[selecter.get_support()].tolist()
    # print(slotOnehotFea)
    # # exit()
    # originDf = pd.concat([originDf, onehotCsr[slotOnehotFea]], axis=1)
    # onehotFea.extend(slotOnehotFea)

    originDf = labelEncoding(originDf, cateFea)
    fea = cateFea + numFea + onehotFea
    # print(originDf[cateFea+numFea].info())
    print("feature prepare: finished!")

    # 划分数据集
    originX = sparse.hstack([sparse.csr_matrix(originDf[fea].astype(float)), tagsMatrix], 'csr').astype('float32')
    dfX = originX[originDf[originDf.flag>=0].index]
    dfy = originDf[originDf.flag>=0]['click']
    trainX = originX[originDf[(originDf.flag>=0)&(originDf.day<6)].index]
    trainy = originDf[(originDf.flag>=0)&(originDf.day<6)]['click']
    validX = originX[originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0).index]
    validy = originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0)['click']
    testX = originX[originDf[originDf.flag==-1].index]
    print('dataset prepare: finished!')

    # 训练模型
    model = LgbModel(fea+tagFea)#, cateFea=onehotFea
    # model.gridSearch(trainX, trainy, validX, validy)
    model.cv(dfX, dfy, nfold=5)
    iterNum = model.train(trainX, trainy, validX=validX, validy=validy)
    model.train(dfX, dfy, num_round=iterNum, verbose=False)

    # skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # # testy = np.zeros((validX.shape[0], skf.n_splits))
    # # for index, (trainIdx, testIdx) in enumerate(skf.split(trainX, trainy)):
    # #     iterNum = model.train(trainX[trainIdx], trainy[trainIdx], validX=trainX[testIdx], validy=trainy[testIdx])
    # #     testy[:, index] = model.predict(validX)
    # # testy = testy.mean(axis=1)
    # # print('valid loss:', metrics.log_loss(validy, testy))
    # testy = np.zeros((testX.shape[0], skf.n_splits))
    # best_score = []
    # for index, (trainIdx, testIdx) in enumerate(skf.split(dfX, dfy)):
    #     iterNum = model.train(dfX[trainIdx], dfy[trainIdx], validX=dfX[testIdx], validy=dfy[testIdx])
    #     testy[:, index] = model.predict(testX)
    #     best_score.append(metrics.log_loss(dfy[testIdx], model.predict(dfX[testIdx])))
    # print('valid score:', best_score, np.mean(best_score))
    # testy = testy.mean(axis=1)
    model.feaScore()
    # exit()

    # 预测结果
    predictDf = originDf[originDf.flag==-1][['instance_id','hour']]
    predictDf['predicted_score'] = model.predict(testX)
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    print(predictDf.groupby('hour')['predicted_score'].mean())
    exportResult(predictDf[['instance_id','predicted_score']], "../result/lgb2_add_num.csv")

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
