#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
from scipy import sparse
import csv
from datetime import *
import json, random, os, math

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
            # 'objective': 'rmse',
            'metric': 'custom',
            'learning_rate': 0.02,
        	'num_leaves': 80,#150
            'max_depth': -1,
            'min_data_in_leaf': 100,#180
            # 'feature_fraction': 0.9,
            'bagging_fraction': 0.95,
        	'bagging_freq': 1,
            'verbose': 0,
            'seed': 0,
        }
        self.params.update(**params)
        self.feaName = feaName
        self.cateFea = cateFea

    def customObj(self, preds, train_data, q=0.3):
        labels = train_data.get_label()
        clicks = (labels>0).astype(int)
        ffms = clicks - labels
        predicts = sigmod(sigmod(ffms,arc=True) + preds)
        grad = predicts - clicks
        hess = predicts * (1.0 - predicts)
        return grad, hess

    def customEval(self, preds, train_data):
        labels = train_data.get_label()
        clicks = (labels>0).astype(int)
        ffms = clicks - labels
        predicts = addupDiff(ffms, preds)
        # predicts =  addupDiff(ffms, preds)
        loss = metrics.log_loss(clicks, predicts)
        return "logloss", loss, False

    def train(self, X, y, num_round=8000, validX=None, validy=None, early_stopping=10, verbose=True, params={}):
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        trainParam = self.params
        trainParam.update(params)
        if validX is not None:
            validData = trainData.create_valid(validX, label=validy)
            bst = lgb.train(trainParam, trainData, num_boost_round=num_round, valid_sets=[trainData,validData], valid_names=['train', 'valid'], fobj=self.customObj, feval=self.customEval, early_stopping_rounds=early_stopping, verbose_eval=verbose)
        else:
            bst = lgb.train(trainParam, trainData, valid_sets=trainData, fobj=self.customObj, feval=self.customEval, num_boost_round=num_round, verbose_eval=verbose)#
        self.bst = bst
        return bst.best_iteration

    def cv(self, X, y, nfold=5, num_round=8000, early_stopping=10, verbose=True, params={}):
        trainParam = self.params
        trainParam.update(params)
        trainData = lgb.Dataset(X, label=y, feature_name=self.feaName, categorical_feature=self.cateFea)
        result = lgb.cv(trainParam, trainData, feature_name=self.feaName, categorical_feature=self.cateFea, num_boost_round=num_round, nfold=nfold, fobj=self.customObj, feval=self.customEval, stratified=False, early_stopping_rounds=early_stopping, verbose_eval=verbose)
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
            # 'num_leaves': [20*i for i in range(2,15)],
            # 'max_depth': list(range(8,13)),
            'min_data_in_leaf': [50*i for i in range(1,10)],
            'bagging_fraction': [1-0.05*i for i in range(0,5)],
            'bagging_freq': list(range(0,10)),

        }
        def getEval(params):
            iter = self.train(X, y, validX=validX, validy=validy, params=params, verbose=verbose)
            return self.bst.best_score['valid'], self.bst.best_iteration
        for k,v in paramsGrids.items():
            resultDf = pd.DataFrame({k: v})
            resultDf['metric_mean'] = list(map(lambda x: getEval({k: x}), v))
            print(resultDf)
        exit()

def sigmod(x, arc=False):
    if arc:
        return -np.log(1/x - 1)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def addupDiff(origins, diffs):
    '''
    根据原始预测值及预测差值返回最终预测值（修正至0-1间）
    '''
    # preds = origins + diffs
    # preds[preds<=0] = (origins[preds<=0]) / 2
    # preds[preds>=1] = (1 + origins[preds>=1]) / 2
    preds = sigmod(sigmod(origins,arc=True) + diffs)
    return preds

def main():
    ORIGIN_DATA_PATH = "../data/"
    FEA_PATH = "../temp/fea2.csv"
    FFM_TRAIN_PATH = "../temp/fusai_keda_nffm_stacking_train.csv"
    FFM_TEST_PATH = "../temp/fusai_keda_nffm_stacking_test.csv"

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

    # 获取ffm结果并计算残差
    if (not os.path.isfile(FFM_TRAIN_PATH)) or (not os.path.isfile(FFM_TEST_PATH)):
        print("[ERROR] could not find ffm stacking file!")
        exit()
    ffmTrainDf = pd.read_csv(FFM_TRAIN_PATH)
    ffmTestDf = pd.read_csv(FFM_TEST_PATH)
    ffmDf = pd.concat([ffmTrainDf,ffmTestDf], ignore_index=True)
    originDf = originDf.merge(ffmDf.rename(columns={'predicted_score': 'ffm_predict'}), how='left', on='instance_id')
    originDf['diff_ffm_label'] = originDf['click'] - originDf['ffm_predict']
    print("load ffm result: finished!")

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
    fea = numFea + cateFea
    print("model dataset prepare: finished!")

    # 划分数据集
    dfX = originDf[originDf.flag>=0][fea].values
    dfy = originDf[originDf.flag>=0]['diff_ffm_label'].values
    trainX = originDf[(originDf.flag>=0)&(originDf.day<6)][fea].values
    trainy = originDf[(originDf.flag>=0)&(originDf.day<6)]['diff_ffm_label'].values
    validX = originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0)[fea].values
    validy = originDf[(originDf.flag>=0)&(originDf.day==6)].sample(frac=0.7, random_state=0)['diff_ffm_label'].values
    testX = originDf[originDf.flag==-1][fea].values
    print('training dataset prepare: finished!')

    # 训练模型
    print('原始logloss:', metrics.log_loss((validy>0).astype(int), (validy>0).astype(int) - validy))
    model = LgbModel(fea)
    # model.gridSearch(trainX, trainy, validX, validy)
    model.cv(dfX, dfy, nfold=5)
    iterNum = model.train(trainX, trainy, validX=validX, validy=validy)
    model.train(dfX, dfy, num_round=iterNum, verbose=False)
    model.feaScore()
    # exit()

    # 预测结果
    modelName = "ffm_lgb_fobj"
    predictDf = originDf[originDf.flag==-1][['instance_id','hour','ffm_predict']]
    predictDf['predict_diff'] = model.predict(testX)
    predictDf['predicted_score'] = predictDf['ffm_predict'] + predictDf['predict_diff']
    # print('invalid predict:\n:', predictDf[(predictDf.predicted_score>=1)|(predictDf.predicted_score<=0)])
    predictDf['predicted_score'] = addupDiff(predictDf['ffm_predict'].values, predictDf['predict_diff'].values)
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    print(predictDf.groupby('hour')['predicted_score'].mean())
    exportResult(predictDf[['instance_id','predicted_score']], "../result/%s.csv"%modelName)

    # 5折stacking
    print('training oof...')
    df2 = originDf[originDf.flag>=0][['instance_id','hour','click','ffm_predict']]
    df2['predict_diff'], predictDf['predict_diff'] = getOof(model, dfX, dfy, testX, stratify=False)
    df2['predicted_score'] = df2['ffm_predict'] + df2['predict_diff']
    # print('invalid predict:\n:', len(df2[(df2.predicted_score>=1)|(df2.predicted_score<=0)]))
    predictDf['predicted_score'] = predictDf['ffm_predict'] + predictDf['predict_diff']
    # print('invalid predict:\n:', predictDf[(predictDf.predicted_score>=1)|(predictDf.predicted_score<=0)])
    print('cv5 valid loss:', metrics.log_loss(df2['click'], df2['predicted_score']))
    print(predictDf[['instance_id','predicted_score']].describe())
    print(predictDf[['instance_id','predicted_score']].head())
    print(predictDf.groupby('hour')['predicted_score'].mean())
    exportResult(df2[['instance_id','predicted_score']], "../result/%s_oof_train.csv"%modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "../result/%s_oof_test.csv"%modelName)
    predictDf['predicted_score'] = addupDiff(predictDf['ffm_predict'].values, predictDf['predict_diff'].values)
    exportResult(predictDf[['instance_id','predicted_score']], "../result/%s_cv.csv"%modelName)


if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print('total time:', datetime.now() - startTime)
