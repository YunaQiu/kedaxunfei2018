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
import json, random, re

from utils import *


def formatDf(df, **params):
    '''
    转换数据类型，统一空缺值，删除无用字段
    '''
    df.drop(['creative_is_js','creative_is_voicead','app_paid','advert_name','creative_is_download','os_name'], axis=1, inplace=True)
    df.loc[df.province==0, ['province','city']] = np.nan
    df.loc[df.os==0, 'os'] = np.nan
    # df = df.astype({
    #     'adid': 'category',
    #     'advert_id': 'category',
    #     'app_cate_id': 'category',
    # })
    return df

def addTime(df, **params):
    '''
    时间字段转换
    已知比赛数据按天分割。若以训练集第一条数据分割，测试集不在同一天。故选择以测试集第一条数据分割
    '''
    dtMin = df[df.flag==-1]['time'].min() - (60*60*24*7)
    df['day'] = (df['time'] - dtMin) // (60*60*24)
    df['hour'] = (df['time'] - dtMin) // (60*60) - (df['day']*24)
    df['date'] = df['day'].astype(str) + "." + df['hour'].map(lambda x: "%02d" % x)
    return df

def addFirstIndustry(df, **params):
    '''
    拆分行业一级和二级
    '''
    df['advert_industry_inner1'] = df['advert_industry_inner'].map(lambda x: x.split("_")[0])
    return df

def addCreativeArea(df, **params):
    '''
    合并创意的宽高
    '''
    df['creative_area'] = df['creative_width'].astype(str) + "_" + df['creative_height'].astype(str)
    return df

def addAppNullCate(df, **params):
    '''
    将APP类型空值作为一个特征值
    '''
    df['app_cate_id'].fillna(0, inplace=True)
    return df

def addSlotPrefix(df, **params):
    '''
    拆分广告位前缀
    '''
    df['slot_prefix'] = df['inner_slot_id'].map(lambda x: x.split('_')[0])
    return df

def addNntType(df, **params):
    '''
    拆分细化联网状态
    '''
    df['is_wifi'] = (df['nnt'] == 1)
    df['nnt_gtype'] = df['nnt'].copy()
    df.loc[(df.nnt < 2) | (df.nnt > 4), 'nnt_gtype'] = np.nan
    return df

def addMajorOsv(df, **params):
    '''
    拆分出系统主版本号
    '''
    startTime = datetime.now()
    df['osv1'] = df['osv'].dropna().astype(str).map(lambda x: re.findall('[0-9]+', x)[0] if len(re.findall('[1-9]+', x))>0 else np.nan)
    df['ios_osv1'] = df.loc[df.os==1, 'osv1']
    df['android_osv1'] = df.loc[df.os==2, 'osv1']
    print('major osv time:', datetime.now() - startTime)
    return df

def addTagsPrefix(df, **params):
    '''
    分割用户标签及用户标签前缀
    '''
    startTime = datetime.now()
    df['tags_list'] = df['user_tags'].dropna().map(lambda x:list(set(x.split(",")) - set([""])))
    df['tags21'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^21", t))>0])
    df['tags21_len'] = df['tags21'].dropna().map(lambda x: len(x))
    df['tags30'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^30\d{5}$", t))>0])
    df['tags30_len'] = df['tags30'].dropna().map(lambda x: len(x))
    df['tagsLen10'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^\d{10}$", t))>0])
    df['tagsLen10_len'] = df['tagsLen10'].dropna().map(lambda x: len(x))
    df['tagsAg'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^ag", t))>0])
    df['tagsAg_len'] = df['tagsAg'].dropna().map(lambda x: len(x))
    df['tagsGd'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^gd", t))>0])
    df['tagsGd_len'] = df['tagsGd'].dropna().map(lambda x: len(x))
    df['tagsMz'] = df['tags_list'].dropna().map(lambda x: [t for t in x if len(re.findall("^mz", t))>0])
    df['tagsMz_len'] = df['tagsMz'].dropna().map(lambda x: len(x))
    print('tags prefix time:', datetime.now() - startTime)
    return df

def addPublishDay(df, **params):
    '''
    广告首次上线时间，已上线时间
    '''
    tempDf = pd.pivot_table(df, index='adid', values='day', aggfunc=np.min)
    df = df.merge(tempDf.rename(columns={'day':'publish_day'}), how='left', left_on='adid', right_index=True)
    df['online_days'] = df['day'] - df['publish_day']
    return df

def addAdvertShow(df, **params):
    '''
    广告主累计出现的曝光数
    '''
    startTime = datetime.now()
    df['advert_showed'] = 1
    tempDf = df.sort_values('time').groupby('advert_id').apply(lambda x: x[['advert_showed']].cumsum())
    df['advert_showed'] = tempDf['advert_showed']
    print('advert showed time:', datetime.now() - startTime)
    return df

def addCreativeAreaRatio(df, **params):
    '''
    创意尺寸的点击率
    '''
    tempDf = df.groupby(['creative_area','day'])['click'].agg([len,'sum'])
    df = df.merge(tempDf[['len']].rename(columns={'len':'creative_area_today_num'}), how='left', left_on=['creative_area','day'], right_index=True)
    tempDf = tempDf.unstack('creative_area').drop([7]).fillna(0).cumsum().stack('creative_area')
    tempDf['creative_area_his_ctr'] = tempDf['sum'] / tempDf['len']
    tempDf = tempDf.reset_index()
    tempDf['creative_area_his_ctr_bias'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'])).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_area','creative_area_his_ctr','creative_area_his_ctr_bias']], how='left', on=['day','creative_area'])
    return df

def addCreativeRatio(df, **params):
    '''
    创意id的点击率
    '''
    tempDf = df.groupby(['day','creative_id'])['click'].agg([len,'sum'])
    df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'creative_today_num'}), how='left', on=['day','creative_id'])
    tempDf = tempDf.unstack('creative_id').drop([7]).fillna(0).cumsum().stack('creative_id').reset_index()
    tempDf['creative_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'])).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_id','creative_his_ctr']], how='left', on=['day','creative_id'])
    return df

def addSlotRatio(df, **params):
    '''
    广告位的点击率
    '''
    tempDf = df.groupby(['day','inner_slot_id'])['click'].agg([len,'sum'])
    df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'slot_today_num'}), how='left', on=['day','inner_slot_id'])
    tempDf = tempDf.unstack('inner_slot_id').drop([7]).fillna(0).cumsum().stack('inner_slot_id').reset_index()
    tempDf['slot_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'])).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','inner_slot_id','slot_his_ctr']], how='left', on=['day','inner_slot_id'])
    return df

def addSlotDnfRatio(df, **params):
    '''
    广告位下的创意id点击率
    '''
    tempDf = df.groupby(['day','inner_slot_id','creative_tp_dnf'])['click'].agg([len,'sum'])
    tempDf = tempDf.unstack(['inner_slot_id','creative_tp_dnf']).fillna(0).cumsum().stack(['inner_slot_id','creative_tp_dnf']).reset_index()
    epsilon = 1 / tempDf['len'].max() ** 2
    tempDf['slot_dnf_his_ctr'] = tempDf.groupby(['day','inner_slot_id']).apply(lambda x: biasSmooth(x['sum'], x['len'], epsilon=epsilon)).reset_index(level=[0,1], drop=True)
    print(tempDf)
    exit()
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','inner_slot_id','slot_his_ctr']], how='left', on=['day','inner_slot_id'])
    exit()

# 城市的点击率

def feaFactory(df):
    startTime = datetime.now()
    df = formatDf(df)
    df = addTime(df)
    df = addPublishDay(df)
    df = addFirstIndustry(df)
    df = addCreativeArea(df)
    df = addAppNullCate(df)
    df = addSlotPrefix(df)
    df = addMajorOsv(df)
    df = addNntType(df)
    df = addAdvertShow(df)
    df = addCreativeAreaRatio(df)
    df = addCreativeRatio(df)
    df = addSlotRatio(df)
    # df = addSlotDnfRatio(df)
    df = addTagsPrefix(df)
    print('feaFactory time:', datetime.now() - startTime)
    return df

if __name__ == '__main__':
    df = importDf("../data/round1_iflyad_train.txt")
    df['flag'] = 0
    predictDf = importDf("../data/round1_iflyad_test_feature.txt")
    predictDf['flag'] = -1
    originDf = pd.concat([df,predictDf], ignore_index=True)
    originDf = feaFactory(originDf)
    exportResult(originDf, "../temp/fea1.csv")
