#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold


# 导入数据
def importDf(url, sep='\t', na_values=None, header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values=na_values, header=header, index_col=index_col, names=colNames)
    return df

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    if isinstance(colName, str):
        colName = [colName]
    colTemp = df[colName]
    df = pd.get_dummies(df, columns=colName)
    df = pd.concat([df, colTemp], axis=1)
    return df

def labelEncoding(df, colList):
    '''
    将标称值转成编码
    '''
    for col in colList:
        df.loc[df[col].notnull(),col] = LabelEncoder().fit_transform(df.loc[df[col].notnull(),col])
        df[col] = df[col].astype(float)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 对数组集合进行合并操作
def listAdd(l):
    result = []
    [result.extend(x) for x in l]
    return result

# 对不同标签进行抽样处理
def getSubsample(labelList, ratio=0.8, repeat=False, params=None):
    if not isinstance(params, dict):
        if isinstance(ratio, (float, int)):
            params = {k:{'ratio':ratio, 'repeat':repeat} for k in set(labelList)}
        else:
            params={k:{'ratio':ratio[k], 'repeat':repeat} for k in ratio.keys()}
    resultIdx = []
    for label in params.keys():
        param = params[label]
        tempList = np.where(labelList==label)[0]
        sampleSize = np.ceil(len(tempList)*params[label]['ratio']).astype(int)
        if (~param['repeat'])&(param['ratio']<=1):
            resultIdx.extend(random.sample(tempList.tolist(),sampleSize))
        else:
            resultIdx.extend(tempList[np.random.randint(len(tempList),size=sampleSize)])
    return resultIdx

# 矩估计法计算贝叶斯平滑参数
def countBetaParamByMME(inputArr, epsilon=0):
    EX = inputArr.mean()
    DX = inputArr.var() + epsilon / len(inputArr)  # 加上极小值防止除以0
    alpha = (EX*(1-EX)/DX - 1) * EX
    beta = (EX*(1-EX)/DX - 1) * (1-EX)
    return alpha,beta

# 对numpy数组进行贝叶斯平滑处理
def biasSmooth(aArr, bArr, method='MME', epsilon=0, alpha=None, beta=None):
    ratioArr = aArr / bArr
    if method=='MME':
        if len(ratioArr[ratioArr==ratioArr]) > 1:
            alpha,beta = countBetaParamByMME(ratioArr[ratioArr==ratioArr], epsilon=epsilon)
        else:
            alpha = beta = 0
        # print(alpha+beta, alpha / (alpha+beta))
    resultArr = (aArr+alpha) / (bArr+alpha+beta)
    return resultArr

# 导出预测结果
def exportResult(df, filePath, header=True, index=False, sep=','):
    df.to_csv(filePath, sep=sep, header=header, index=index)

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=True, verbose=False, random_state=0, weight=None):
    startTime = datetime.now()
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    if stratify:
        kf = StratifiedKFold(n_splits=nFold, random_state=random_state, shuffle=True)
    else:
        kf = KFold(n_splits=nFold, random_state=random_state, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX, trainY)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        kfTesty = trainY[testIdx]
        # if weight is not None:
        #     kfWeight = weight[trainIdx]
        # else:
        #     kfWeight = None
        clf.train(kfTrainX, kfTrainY, validX=kfTestX, validy=kfTesty, verbose=verbose)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
        print('oof cv %d of %d: finished!' % (i+1, nFold))
    oofTest[:] = oofTestSkf.mean(axis=1)
    print('oof cost time:', datetime.now() - startTime)
    return oofTrain, oofTest
