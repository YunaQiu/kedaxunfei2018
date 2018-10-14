#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
from scipy import sparse
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile

from utils import *


def formatDf(df, **params):
    '''
    转换数据类型，统一空缺值，删除无用字段
    '''
    df.drop(['creative_is_js','creative_is_voicead','app_paid','advert_name','creative_is_download','os_name'], axis=1, inplace=True)
    df.loc[df.province==0, ['province','city']] = np.nan
    df.loc[df.os==0, 'os'] = np.nan
    df = df.astype({
        'creative_is_jump': 'float',
        'creative_has_deeplink': 'float',
    })
    return df

def addTime(df, **params):
    '''
    时间字段转换
    '''
    df['datetime'] = pd.to_datetime(df['time'], unit='s') + pd.Timedelta(hours=8)
    dateMin = df['datetime'].dt.date.min()
    df['day'] = (df['datetime'] - dateMin).dt.days
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_minute'] = 100*df['hour'] + df['minute']
    return df

def addFirstIndustry(df, **params):
    '''
    拆分行业一级和二级
    '''
    df['advert_industry_inner1'] = df['advert_industry_inner'].map(lambda x: x.split("_")[0])
    df['advert_industry_inner2'] = df['advert_industry_inner'].map(lambda x: x.split("_")[1])
    return df

def addCreativeArea(df, **params):
    '''
    合并创意的宽高
    '''
    df['creative_area'] = df['creative_width'] * df['creative_height']
    df['creative_dpi'] = df['creative_width'].astype(str) + "_" + df['creative_height'].astype(str)
    return df

def addAppNullCate(df, **params):
    '''
    将APP类型空值作为一个特征值
    '''
    df['app_cate_id'].fillna(0, inplace=True)
    return df

def addAppTailNumber(df, **params):
    '''
    提取appid的尾号
    '''
    df['app_tail_number'] = df['app_id'].dropna().map(lambda x: int(x % 10000))
    return df

def addAdcode(df, **params):
    '''
    添加省份及城市编码标识
    '''
    directCityCode = [137101101100100,137101102100100,137103101100100,137105101100100]  # 直辖市
    specialProvinceCode = [137107101100100,137107102100100,137107103100100] # 港澳台
    tempDf = df[['province','city']].dropna().drop_duplicates()
    tempDf['isDirectCity'] = (tempDf.province.isin(directCityCode)).astype(int)
    tempDf['region'] = tempDf['province'].map(lambda x: int(str(x)[5]))
    tempDf['cityCode'] = tempDf['city'].map(lambda x: int(str(x)[10:12]))
    tempDf.loc[tempDf.cityCode==0, 'cityCode'] = np.nan
    df = df.merge(tempDf, how='left', on=['province','city'])
    return df

def addSlotPrefix(df, **params):
    '''
    拆分广告位前缀
    '''
    df['slot_prefix'] = df['inner_slot_id'].map(lambda x: x.split('_')[0])
    df['slot2'] = df['inner_slot_id'].map(lambda x: x.split('_')[1])
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

def splitTagsList(df, **params):
    '''
    分割用户标签
    '''
    startTime = datetime.now()
    df['tags_list'] = df['user_tags'].dropna().map(lambda x:list(set(x.split(",")) - set([""])))
    df['tags_num'] = df['tags_list'].dropna().map(lambda x: len(x))
    df['tags_num'].fillna(0, inplace=True)
    print('split tags time:', datetime.now() - startTime)
    return df

def addTagsPrefix(df, **params):
    '''
    添加用户标签前缀
    '''
    startTime = datetime.now()
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

def addTagsMean(df, **params):
    df['tags21_mean'] = df['tags21'].dropna().map(lambda x: np.array(x).astype(float).mean())
    df['tags30_mean'] = df['tags30'].dropna().map(lambda x: np.array(x).astype(float).mean())
    return df

def addTagsMatrix(df, **params):
    '''
    将用户标签转换成稀疏矩阵
    '''
    startTime = datetime.now()
    cv = CountVectorizer(min_df=0.001, max_df=0.8, binary=True)
    cv.fit(df['user_tags'].dropna())
    tagSelecter = SelectPercentile(chi2, percentile=10)
    tagSelecter.fit(cv.transform(df[df.flag>=0]['user_tags'].fillna("")), df[df.flag>=0]['click'])
    tagsMatrix = tagSelecter.transform(cv.transform(df['user_tags'].fillna("")))
    tagsName = np.array(cv.get_feature_names())[tagSelecter.get_support()]
    tempDf = pd.DataFrame(tagsMatrix.toarray(), columns=['tag_'+str(x) for x in tagsName], index=df.index)
    df = pd.concat([df, tempDf], axis=1)
    print('tag matrix time:', datetime.now() - startTime)
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

def addAdidRatio(df, **params):
    '''
    广告的点击率
    '''
    tempDf = df.groupby(['day','adid'])['click'].agg([len,'sum'])
    # df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'ad_today_num'}), how='left', on=['day','adid'])
    tempDf = tempDf.unstack('adid').drop([7]).fillna(0).cumsum().stack('adid').reset_index()
    epsilon = tempDf['len'].max() ** 2
    tempDf['ad_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'], epsilon=epsilon)).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','adid','ad_his_ctr']], how='left', on=['day','adid'])
    return df

def addColNum(df, colName, prefix, **params):
    tempDf = pd.pivot_table(df, index=['day',colName], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf.loc[7:] *= 4
    tempDf['click_addup'] = tempDf.unstack(colName).cumsum().stack(colName)
    tempDf = tempDf.reset_index(level=colName)
    daySeries = df.groupby('day')['instance_id'].count()
    daySeries.loc[7] *= 4
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['%s_today_num_ratio'%prefix] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day',colName,'%s_today_num_ratio'%prefix]], how='left', on=['day',colName])
    tempDf['%s_his_num_ratio'%prefix] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day',colName,'%s_his_num_ratio'%prefix]], how='left', on=['day',colName])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][[colName,'%s_his_num_ratio'%prefix]].rename(columns={'%s_his_num_ratio'%prefix:'%s_num_ratio'%prefix}), how='left', on=[colName])
    tempDf = tempDf.groupby(colName)[['%s_today_num_ratio'%prefix]].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'%s_today_num_ratio'%prefix:'%s_num_ratio_mean'%prefix}), how='left', on=[colName])
    return df

def addAdidNum(df, **params):
    '''
    广告id当天出现的频率
    '''
    tempDf = pd.pivot_table(df, index=['day','adid'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf.loc[7:] *= 4
    tempDf['click_addup'] = tempDf.unstack('adid').cumsum().stack('adid')
    tempDf = tempDf.reset_index(level='adid')
    daySeries = df.groupby('day')['instance_id'].count()
    daySeries.loc[7] *= 4
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['ad_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','adid','ad_today_num_ratio']], how='left', on=['day','adid'])
    tempDf['ad_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','adid','ad_his_num_ratio']], how='left', on=['day','adid'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['adid','ad_his_num_ratio']].rename(columns={'ad_his_num_ratio':'ad_num_ratio'}), how='left', on=['adid'])
    tempDf = tempDf.groupby('adid')[['ad_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'ad_today_num_ratio':'ad_num_ratio_mean'}), how='left', on=['adid'])
    return df

def addCreativeAreaRatio(df, **params):
    '''
    创意尺寸的点击率
    '''
    tempDf = df.groupby(['creative_area','day'])['click'].agg([len,'sum'])
    # df = df.merge(tempDf[['len']].rename(columns={'len':'creative_area_today_num'}), how='left', left_on=['creative_area','day'], right_index=True)
    tempDf = tempDf.unstack('creative_area').drop([7]).fillna(0).cumsum().stack('creative_area')
    tempDf['creative_area_his_ctr'] = tempDf['sum'] / tempDf['len']
    tempDf = tempDf.reset_index()
    epsilon = tempDf['len'].max() ** 2
    tempDf['creative_area_his_ctr_bias'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'], epsilon=epsilon)).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_area','creative_area_his_ctr','creative_area_his_ctr_bias']], how='left', on=['day','creative_area'])
    return df

def addCreativeDpiNum(df, **params):
    '''
    创意尺寸当天出现的频率
    '''
    # tempDf = df.groupby(['day','creative_dpi'])[['click']].agg(len).reset_index(level='creative_dpi')
    # tempDf['all'] = df.groupby('day')['instance_id'].count()
    # tempDf['dpi_today_num_ratio'] = tempDf['click'] / tempDf['all']
    # df = df.merge(tempDf.reset_index()[['day','creative_dpi','dpi_today_num_ratio']], how='left', on=['day','creative_dpi'])

    tempDf = pd.pivot_table(df, index=['day','creative_dpi'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf['click_addup'] = tempDf.unstack('creative_dpi').cumsum().stack('creative_dpi')
    tempDf = tempDf.reset_index(level='creative_dpi')
    daySeries = df.groupby('day')['creative_dpi'].count()
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['dpi_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','creative_dpi','dpi_today_num_ratio']], how='left', on=['day','creative_dpi'])
    tempDf['dpi_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_dpi','dpi_his_num_ratio']], how='left', on=['day','creative_dpi'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['creative_dpi','dpi_his_num_ratio']].rename(columns={'dpi_his_num_ratio':'dpi_num_ratio'}), how='left', on=['creative_dpi'])
    tempDf = tempDf.groupby('creative_dpi')[['dpi_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'dpi_today_num_ratio':'dpi_num_ratio_mean'}), how='left', on=['creative_dpi'])
    return df

def addCreativeRatio(df, **params):
    '''
    创意id的点击率
    '''
    tempDf = df.groupby(['day','creative_id'])['click'].agg([len,'sum'])
    # df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'creative_today_num'}), how='left', on=['day','creative_id'])
    tempDf = tempDf.unstack('creative_id').drop([7]).fillna(0).cumsum().stack('creative_id').reset_index()
    epsilon = tempDf['len'].max() ** 2
    tempDf['creative_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'], epsilon=epsilon)).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_id','creative_his_ctr']], how='left', on=['day','creative_id'])
    return df

def addCreativeNum(df, **params):
    '''
    创意当天出现的频率
    '''
    # tempDf = df.groupby(['day','creative_id'])[['click']].agg(len).reset_index(level='creative_id')
    # tempDf['all'] = df.groupby('day')['instance_id'].count()
    # tempDf['creative_today_num_ratio'] = tempDf['click'] / tempDf['all']
    # df = df.merge(tempDf.reset_index()[['day','creative_id','creative_today_num_ratio']], how='left', on=['day','creative_id'])

    tempDf = pd.pivot_table(df, index=['day','creative_id'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf['click_addup'] = tempDf.unstack('creative_id').cumsum().stack('creative_id')
    tempDf = tempDf.reset_index(level='creative_id')
    daySeries = df.groupby('day')['creative_id'].count()
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['creative_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','creative_id','creative_today_num_ratio']], how='left', on=['day','creative_id'])
    tempDf['creative_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','creative_id','creative_his_num_ratio']], how='left', on=['day','creative_id'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['creative_id','creative_his_num_ratio']].rename(columns={'creative_his_num_ratio':'creative_num_ratio'}), how='left', on=['creative_id'])
    tempDf = tempDf.groupby('creative_id')[['creative_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'creative_today_num_ratio':'creative_num_ratio_mean'}), how='left', on=['creative_id'])
    return df

def addSlotRatio(df, **params):
    '''
    广告位的点击率
    '''
    tempDf = df.groupby(['day','inner_slot_id'])['click'].agg([len,'sum'])
    # df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'slot_today_num'}), how='left', on=['day','inner_slot_id'])
    tempDf = tempDf.unstack('inner_slot_id').drop([7]).fillna(0).cumsum().stack('inner_slot_id').reset_index()
    epsilon = tempDf['len'].max() ** 2
    tempDf['slot_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'], epsilon=epsilon)).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','inner_slot_id','slot_his_ctr']], how='left', on=['day','inner_slot_id'])
    return df

def addSlotNum(df, **params):
    '''
    广告位当天出现的频率
    '''
    # tempDf = df.groupby(['day','inner_slot_id'])[['click']].agg(len).reset_index(level='inner_slot_id')
    # tempDf['all'] = df.groupby('day')['instance_id'].count()
    # tempDf['slot_today_num_ratio'] = tempDf['click'] / tempDf['all']
    # df = df.merge(tempDf.reset_index()[['day','inner_slot_id','slot_today_num_ratio']], how='left', on=['day','inner_slot_id'])

    tempDf = pd.pivot_table(df, index=['day','inner_slot_id'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf['click_addup'] = tempDf.unstack('inner_slot_id').cumsum().stack('inner_slot_id')
    tempDf = tempDf.reset_index(level='inner_slot_id')
    daySeries = df.groupby('day')['inner_slot_id'].count()
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['slot_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','inner_slot_id','slot_today_num_ratio']], how='left', on=['day','inner_slot_id'])
    tempDf['slot_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','inner_slot_id','slot_his_num_ratio']], how='left', on=['day','inner_slot_id'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['inner_slot_id','slot_his_num_ratio']].rename(columns={'slot_his_num_ratio':'slot_num_ratio'}), how='left', on=['inner_slot_id'])
    tempDf = tempDf.groupby('inner_slot_id')[['slot_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'slot_today_num_ratio':'slot_num_ratio_mean'}), how='left', on=['inner_slot_id'])
    return df

def addAppNum(df, **params):
    '''
    广告位当天出现的频率
    '''
    # tempDf = df.groupby(['day','app_id'])[['click']].agg(len).reset_index(level='app_id')
    # tempDf['all'] = df.groupby('day')['instance_id'].count()
    # tempDf['app_today_num_ratio'] = tempDf['click'] / tempDf['all']
    # df = df.merge(tempDf.reset_index()[['day','app_id','app_today_num_ratio']], how='left', on=['day','app_id'])

    tempDf = pd.pivot_table(df, index=['day','app_id'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf['click_addup'] = tempDf.unstack('app_id').cumsum().stack('app_id')
    tempDf = tempDf.reset_index(level='app_id')
    daySeries = df.groupby('day')['app_id'].count()
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['app_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','app_id','app_today_num_ratio']], how='left', on=['day','app_id'])
    tempDf['app_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','app_id','app_his_num_ratio']], how='left', on=['day','app_id'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['app_id','app_his_num_ratio']].rename(columns={'app_his_num_ratio':'app_num_ratio'}), how='left', on=['app_id'])
    tempDf = tempDf.groupby('app_id')[['app_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'app_today_num_ratio':'app_num_ratio_mean'}), how='left', on=['app_id'])
    return df

def addCityRatio(df, **params):
    '''
    城市的点击率
    '''
    tempDf = df.groupby(['day','city'])['click'].agg([len,'sum'])
    # df = df.merge(tempDf[['len']].reset_index().rename(columns={'len':'city_today_num'}), how='left', on=['day','city'])
    tempDf = tempDf.unstack('city').drop([7]).fillna(0).cumsum().stack('city').reset_index()
    epsilon = tempDf['len'].max() ** 2
    tempDf['city_his_ctr'] = tempDf.groupby('day').apply(lambda x: biasSmooth(x['sum'],x['len'], epsilon=epsilon)).reset_index(level=0, drop=True)
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','city','city_his_ctr']], how='left', on=['day','city'])
    return df

def addCityNum(df, **params):
    '''
    广告位当天出现的频率
    '''
    # tempDf = df.groupby(['day','city'])[['click']].agg(len).reset_index(level='city')
    # tempDf['all'] = df.groupby('day')['instance_id'].count()
    # tempDf['city_today_num_ratio'] = tempDf['click'] / tempDf['all']
    # df = df.merge(tempDf.reset_index()[['day','city','city_today_num_ratio']], how='left', on=['day','city'])

    tempDf = pd.pivot_table(df, index=['day','city'], values=['click'], aggfunc=len, dropna=False, fill_value=0)
    tempDf['click_addup'] = tempDf.unstack('city').cumsum().stack('city')
    tempDf = tempDf.reset_index(level='city')
    daySeries = df.groupby('day')['city'].count()
    tempDf['day_len'] = daySeries
    tempDf['day_addup'] = daySeries.cumsum()
    tempDf = tempDf.reset_index()
    tempDf['city_today_num_ratio'] = tempDf['click'] / tempDf['day_len']
    df = df.merge(tempDf[['day','city','city_today_num_ratio']], how='left', on=['day','city'])
    tempDf['city_his_num_ratio'] = tempDf['click_addup'] / tempDf['day_addup']
    tempDf['day'] += 1
    df = df.merge(tempDf[['day','city','city_his_num_ratio']], how='left', on=['day','city'])
    df = df.merge(tempDf[tempDf.day == tempDf.day.max()][['city','city_his_num_ratio']].rename(columns={'city_his_num_ratio':'city_num_ratio'}), how='left', on=['city'])
    tempDf = tempDf.groupby('city')[['city_today_num_ratio']].mean().reset_index()
    df = df.merge(tempDf.rename(columns={'city_today_num_ratio':'city_num_ratio_mean'}), how='left', on=['city'])
    return df

def addCrossColNum(df, col1, col2, alias, **params):
    '''
    增加交叉特征的独立个数
    '''
    tempDf = pd.pivot_table(df, index=col1, values=col2, aggfunc='nunique')[[col2]]
    tempDf = tempDf.rename(columns={col2:'%s_nunique'%alias}).reset_index()
    df = df.merge(tempDf, how='left', on=[col1])
    return df

def feaFactory(df):
    startTime = datetime.now()
    df = formatDf(df)
    df = addTime(df)
    df = addPublishDay(df)
    df = addFirstIndustry(df)
    df = addCreativeArea(df)
    df = addAppTailNumber(df)
    df = addAppNullCate(df)
    df = addAdcode(df)
    df = addSlotPrefix(df)
    df = addMajorOsv(df)
    df = addNntType(df)
    df = addAdvertShow(df)
    df = addAdidRatio(df)
    df = addCreativeAreaRatio(df)
    df = addCreativeRatio(df)
    df = addSlotRatio(df)
    df = addCityRatio(df)
    colNumList = [
        ["adid", "ad"],
        ["creative_dpi", "dpi"],
        ["creative_id", "creative"],
        ["inner_slot_id", "slot"],
        ["app_id", "app"],
        ["city", "city"],
        ["model", "model"],
        ["make", "make"],
        ['advert_id', 'advert'],
        ['advert_industry_inner', 'industry'],
        ['campaign_id', 'campaign'],
    ]
    for k,v in colNumList:
        df = addColNum(df, k, v)
    crossColList = [
        ['creative_id', 'adid', 'creative_ad'],
        ['adid', 'app_id', 'ad_app'],
        ['adid', 'inner_slot_id', 'ad_slot'],
        ['inner_slot_id', 'adid', 'slot_ad'],
        ['inner_slot_id', 'creative_id', 'slot_creative'],
        ['campaign_id', 'orderid', 'campaign_order'],
        ['campaign_id', 'creative_id', 'campaign_creative'],
        ['app_id','inner_slot_id','app_slot'],
        ['model','creative_dpi', 'model_dpi'],
        ['orderid','adid','order_ad'],
    ]
    for c1,c2,alias in crossColList:
        df = addCrossColNum(df, c1, c2, alias)
    df = splitTagsList(df)
    # df = addTagsMatrix(df)
    df = addTagsPrefix(df)
    df = addTagsMean(df)
    df.index = list(range(len(df)))
    print('feaFactory time:', datetime.now() - startTime)
    return df

def userTagsMatrix(tagSeries):
    '''
    将用户标签转换成稀疏矩阵
    '''
    startTime = datetime.now()
    cv = CountVectorizer(min_df=0.001, max_df=0.8, binary=True)
    cv.fit(tagSeries.dropna())
    tagCsr = cv.transform(tagSeries.fillna(""))
    tagName = np.array(cv.get_feature_names())
    print('user tag time:', datetime.now() - startTime)
    return tagCsr, tagName

if __name__ == '__main__':
    df1 = importDf("../data/round1_iflyad_train.txt")
    df1['flag'] = 1
    df2 = importDf("../data/round2_iflyad_train.txt")
    df2['flag'] = 2
    df = pd.concat([df1, df2], ignore_index=True).sample(frac=0.1, random_state=0)
    df.drop_duplicates(subset=['instance_id'], inplace=True)
    predictDf = importDf("../data/round2_iflyad_test_feature.txt")
    predictDf['flag'] = -1
    originDf = pd.concat([df,predictDf], ignore_index=True)
    originDf = feaFactory(originDf)
    exportResult(originDf, "../temp/fea2.csv")

    tagCsr, tagName = userTagsMatrix(originDf['user_tags'])
    sparse.save_npz("../temp/user_tags2.npz", tagCsr)
    fp = open("../temp/user_tags_name2.txt", "w")
    fp.write(",".join(tagName))
    fp.close()
