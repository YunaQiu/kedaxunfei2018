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
from feature2 import feaFactory, userTagsMatrix


class LgbModel:
    def __init__(self, feaName, cateFea=[], params={}):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
        	'num_leaves': 150,
            'max_depth': -1,
            'min_data_in_leaf': 350,
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
        	'bagging_freq': 3,
            'verbose': 0,
            'seed': 0,
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
            'num_leaves': [20*i for i in range(2,10)],
            # 'max_depth': list(range(8,13)),
            # 'min_data_in_leaf': [50*i for i in range(2,10)],
            # 'bagging_fraction': [1-0.05*i for i in range(0,5)],
            # 'bagging_freq': list(range(0,10)),

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
    ORIGIN_DATA_PATH = "../data/"
    FEA_PATH = "../temp/fea2.csv"
    TAGS_PATH = "../temp/user_tags2.npz"
    TAGS_NAME_PATH = "../temp/user_tags_name2.txt"
    SPARSE_COL_PATH = "../temp/lgb2_sparse_col.npz"
    SPARSE_COLNAME_PATH = "../temp/lgb2_sparse_colname.txt"

    # 获取特征工程数据集
    if not os.path.isfile(FEA_PATH):
        df1 = importDf(ORIGIN_DATA_PATH + "round1_iflyad_train.txt")
        df1['flag'] = 1
        df2 = importDf(ORIGIN_DATA_PATH + "round2_iflyad_train.txt")
        df2['flag'] = 2
        df1.drop(df1[df1.instance_id.isin(df2.instance_id)].index, inplace=True)
        predictDf = importDf(ORIGIN_DATA_PATH + "round2_iflyad_test_feature.txt")
        predictDf['flag'] = -1
        predictDf2 = importDf(ORIGIN_DATA_PATH + "round1_iflyad_test_feature.txt")
        predictDf2.drop(predictDf2[predictDf2.instance_id.isin(predictDf.instance_id)].index, inplace=True)
        predictDf2['flag'] = -2
        originDf = pd.concat([df1, df2, predictDf, predictDf2], ignore_index=True)
        originDf = feaFactory(originDf)
        exportResult(originDf, FEA_PATH)
    else:
        originDf = pd.read_csv(FEA_PATH)
    print("feature dataset prepare: finished!")


    # 筛选稀疏特征
    if not os.path.isfile(SPARSE_COL_PATH):
        if not os.path.isfile(TAGS_PATH):
            sparseCsr, sparseFea = userTagsMatrix(originDf['user_tags'])
            sparse.save_npz(TAGS_PATH, sparseCsr)
            fp = open(TAGS_NAME_PATH, "w")
            fp.write(",".join(sparseFea))
            fp.close()
        else:
            sparseCsr = sparse.load_npz(TAGS_PATH)
            fp = open(TAGS_NAME_PATH)
            sparseFea = None
            try:
                sparseFea = fp.read().split(",")
            finally:
                fp.close()
        sparseFea = ["tag_%s"%x for x in sparseFea]
        selecter = SelectPercentile(chi2, percentile=20)
        selecter.fit(sparseCsr[originDf[originDf.flag>=0].index.tolist()], originDf[originDf.flag>=0]['click'])
        sparseCsr = selecter.transform(sparseCsr)
        sparseFea = np.array(sparseFea)[selecter.get_support()].tolist()
        print("%d tags fea select: finished!" % len(sparseFea))

        onehotList = ['creative_type','creative_dpi','advert_industry_inner1','slot_prefix','region','carrier','nnt','devtype','os']
        onehotDf = pd.get_dummies(originDf[onehotList], columns=onehotList, sparse=True)
        sparseCsr = sparse.hstack([sparseCsr, sparse.csr_matrix(onehotDf)], 'csr')
        # print(onehotDf.columns[:5])
        sparseFea.extend(onehotDf.columns)
        print("onehot fea:", len(onehotDf.columns))

        selectList = ['adid','inner_slot_id','make','creative_id','app_id']
        for x in selectList:
            onehotDf = pd.get_dummies(originDf[[x]], columns=[x], sparse=True)
            onehotCsr = sparse.csr_matrix(onehotDf)
            selecter = SelectPercentile(percentile=5)
            selecter.fit(onehotCsr[originDf[originDf.flag>=0].index.tolist()], originDf[originDf.flag>=0]['click'])
            sparseCsr = sparse.hstack([sparseCsr, selecter.transform(onehotCsr)], 'csr')
            sparseFea.extend(onehotDf.columns[selecter.get_support()])
            print('select %s onehot: %d' % (x, len(selecter.get_support(indices=True))))

        sparse.save_npz(SPARSE_COL_PATH, sparseCsr)
        fp = open(SPARSE_COLNAME_PATH, "w")
        fp.write("||".join(sparseFea).replace(" ","_"))
        fp.close()
    else:
        sparseCsr = sparse.load_npz(SPARSE_COL_PATH)
        fp = open(SPARSE_COLNAME_PATH)
        sparseFea = np.array(list(range(sparseCsr.shape[1]))).astype(str).tolist()
        try:
            sparseFea = fp.read().replace(" ","_").split("||")
        finally:
            fp.close()
    print("sparse dataset prepare: finished!")

    # 全部特征拼接
    cateFea = [
        'adid','advert_id','orderid','advert_industry_inner','campaign_id','creative_id','creative_type','creative_tp_dnf',
        'app_cate_id','f_channel','app_id','inner_slot_id',
        'city','province','carrier','nnt','devtype','os','osv','make','model',

        'advert_industry_inner1','creative_dpi',
        'slot_prefix',#'slot2',
        'region',
        ]
    numFea = [
        'creative_width','creative_height','creative_has_deeplink','creative_is_jump',

        'hour','hour_minute','minute',#'online_days','day',
        'creative_area',#'creative_his_ctr',#'ad_his_ctr','creative_area_his_ctr',# 'advert_showed',#'ad_today_num','creative_area_today_num','creative_today_num',
        'app_tail_number',#'slot_his_ctr',# 'slot_today_num',
        'cityCode','osv1','ios_osv1','android_osv1','isDirectCity','nnt_gtype',#'city_his_ctr',#'city_today_num','is_wifi',
        'tags_num','tags21_len','tags30_len','tagsLen10_len','tagsAg_len','tagsGd_len','tagsMz_len',#'tags21_mean','tags30_mean',
        # 'ad_today_num_ratio','dpi_today_num_ratio','creative_today_num_ratio','slot_today_num_ratio','app_today_num_ratio','city_today_num_ratio',
        'ad_num_ratio','dpi_num_ratio','creative_num_ratio','slot_num_ratio','app_num_ratio','city_num_ratio','make_num_ratio','model_num_ratio',#'advert_num_ratio','industry_num_ratio','campaign_num_ratio',
        'creative_ad_nunique','app_slot_nunique','model_dpi_nunique','order_ad_nunique','slot_ad_nunique','slot_creative_nunique','ad_app_nunique','ad_slot_nunique',#'campaign_order_nunique','campaign_creative_nunique',
        ]
    originDf = labelEncoding(originDf, cateFea)
    fea = cateFea + numFea
    originX = sparse.hstack([sparse.csr_matrix(originDf[fea].astype(float)), sparseCsr], 'csr').astype('float32')
    fea.extend(sparseFea)
    print('model dataset size:', originX.shape)
    print("model dataset prepare: finished!")

    # 划分数据集
    dfX = originX[originDf[originDf.flag>=0].index.tolist()]
    dfy = originDf[originDf.flag>=0]['click']
    trainX = originX[originDf[(originDf.flag>=0)&(originDf.day<6)].index.tolist()]
    trainy = originDf[(originDf.flag>=0)&(originDf.day<6)]['click']
    validX = originX[originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0).index.tolist()]
    validy = originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0)['click']
    testX = originX[originDf[originDf.flag==-1].index.tolist()]
    print('training dataset prepare: finished!')

    # 训练模型
    model = LgbModel(fea)
    model.gridSearch(trainX, trainy, validX, validy)
    model.cv(dfX, dfy, nfold=5)
    iterNum = model.train(trainX, trainy, validX=validX, validy=validy, params={'learning_rate':0.02})
    model.train(dfX, dfy, num_round=iterNum, params={'learning_rate':0.02}, verbose=False)
    model.feaScore()

    # 预测结果
    predictDf = originDf[originDf.flag==-1][['instance_id','hour']]
    predictDf['predicted_score'] = model.predict(testX)
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    print(predictDf.groupby('hour')['predicted_score'].mean())
    exportResult(predictDf[['instance_id','predicted_score']], "../result/lgb2.csv")

    # 5折stacking
    print('training oof...')
    df2 = originDf[originDf.flag>=0][['instance_id','hour','click']]
    df2['predicted_score'], predictDf['predicted_score'] = getOof(model, dfX, dfy, testX)
    print('cv5 valid loss:', metrics.log_loss(df2['click'], df2['predicted_score']))
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    print(predictDf.groupby('hour')['predicted_score'].mean())
    exportResult(df2[['instance_id','predicted_score']], "../result/lgb2_oof_train.csv")
    exportResult(predictDf[['instance_id','predicted_score']], "../result/lgb2_oof_test.csv")


if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
