#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 可视化特征
"""

import os
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt

glo_log_dir = ""


def plot_and_save_histogram(pos_values, neg_values, feature_dir, title):
    """
    绘制直方图
    :param pos_values: 正样本特征
    :param neg_values: 负样本特征
    :param feature_dir: 特征保存目录
    :param title: 直方图标题
    :return:
    """
    pos_values = np.array(pos_values)
    neg_values = np.array(neg_values)

    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)
    plt.hist(pos_values, **kwargs, label='attack', density=True)
    plt.hist(neg_values, **kwargs, label='clean', density=True)
    plt.legend()
    plt.title(title)

    old_filename = os.path.join(feature_dir, f"{title}.png")
    plt.savefig(old_filename)
    new_filename = os.path.join(glo_log_dir, f"{title}.log")
    shutil.copy2(old_filename, new_filename)
    plt.close()


def keyword_match(keyword_features, feature_dir):
    """
    分析关键词匹配特征keyword_match，可视化
    :param keyword_features: 关键词特征
    :param feature_dir: 特征保存目录
    :return:
    """
    keyword_match_pos, keyword_match_neg = [], []
    for keyword_feature in keyword_features:
        keyword_match_ = keyword_feature['feature']['keyword_match']
        label = keyword_feature['label']
        if label > 0:
            keyword_match_pos.append(keyword_match_)
        else:
            keyword_match_neg.append(keyword_match_)

    plot_and_save_histogram(keyword_match_pos, keyword_match_neg, feature_dir, "keyword_match")


def bayes_prob(keyword_features, feature_dir):
    """
    分析贝叶斯概率bayes_prob，可视化
    :param keyword_features: 关键词特征
    :param feature_dir: 特征保存目录
    :return:
    """
    bayes_prob_pos, bayes_prob_neg = [], []
    for keyword_feature in keyword_features:
        if 'bayes_prob' not in keyword_feature['feature']:
            continue
        bayes_prob_ = keyword_feature['feature']['bayes_prob']
        label = keyword_feature['label']
        if label > 0:
            bayes_prob_pos.append(bayes_prob_)
        else:
            bayes_prob_neg.append(bayes_prob_)

    plot_and_save_histogram(bayes_prob_pos, bayes_prob_neg, feature_dir, "bayes_prob")


def keyword_dist_mean(keyword_features, feature_dir):
    """
    分析关键词距离均值keyword_dist_mean，可视化
    :param keyword_features: 关键词特征
    :param feature_dir: 特征保存目录
    :return:
    """
    keyword_dist_mean_pos, keyword_dist_mean_neg = [], []
    for keyword_feature in keyword_features:
        label = keyword_feature['label']
        keyword_feature = keyword_feature['feature']
        if 'dist_feature' in keyword_feature:

            if label > 0:
                keyword_dist_mean_pos.append(keyword_feature['dist_feature']['mean'])
            else:
                keyword_dist_mean_neg.append(keyword_feature['dist_feature']['mean'])

    plot_and_save_histogram(keyword_dist_mean_pos, keyword_dist_mean_neg, feature_dir, "keyword_dist_mean")


def keyword_dist_qt(logger, keyword_features, feature_dir):
    """
    分析关键词距离四分位数keyword_dist_qt，可视化
    :param logger: 日志对象
    :param keyword_features: 关键词特征
    :param feature_dir: 特征保存目录
    :return:
    """
    keyword_dist_qt_pos, keyword_dist_qt_neg = [], []
    for keyword_feature in keyword_features:
        label = keyword_feature['label']
        keyword_feature = keyword_feature['feature']
        if 'dist_feature' in keyword_feature:

            if label > 0:
                keyword_dist_qt_pos.append(keyword_feature['dist_feature']['qt'])
            else:
                keyword_dist_qt_neg.append(keyword_feature['dist_feature']['qt'])

    # 补充逻辑：如果 keyword_dist_qt_pos 或 keyword_dist_qt_neg 为空，直接返回，否则下面画直方图会出错
    if not keyword_dist_qt_pos or not keyword_dist_qt_neg:
        logger.warning("keyword_dist_qt_pos or keyword_dist_qt_neg is empty, cannot plot keyword_dist_qt.png")
        return

    keyword_dist_qt_pos, keyword_dist_qt_neg = np.array(keyword_dist_qt_pos), np.array(keyword_dist_qt_neg)
    keyword_dist_qt_pos = keyword_dist_qt_pos.transpose()
    keyword_dist_qt_neg = keyword_dist_qt_neg.transpose()

    m = len(keyword_dist_qt_pos)
    fig, axs = plt.subplots(m, 1, figsize=(20, 20*m))

    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)
    for i in range(m):
        axs[i].hist(keyword_dist_qt_pos[i, :], **kwargs, label='attack', density=True)
        axs[i].hist(keyword_dist_qt_neg[i, :], **kwargs, label='clean', density=True)
        axs[i].set_title('qt %d' % (i))
        axs[i].legend()
    plt.title('keyword_dist_qt')

    old_filename = os.path.join(feature_dir, 'keyword_dist_qt.png')
    plt.savefig(old_filename)
    new_filename = os.path.join(glo_log_dir, "keyword_dist_qt.log")
    shutil.copy2(old_filename, new_filename)
    plt.close()


def doc_match(doc_features, feature_dir):
    """
    分析攻击手法匹配doc_match，可视化
    :param doc_features: 攻击手法特征
    :param feature_dir: 特征保存目录
    :return:
    """
    doc_match_pos, doc_match_neg = [], []
    for doc_feature in doc_features:
        doc_match_ = doc_feature['feature']['doc_match']
        label = doc_feature['label']
        if label > 0:
            doc_match_pos.append(doc_match_)
        else:
            doc_match_neg.append(doc_match_)

    plot_and_save_histogram(doc_match_pos, doc_match_neg, feature_dir, "doc_match")


def weight_sim(doc_features, feature_dir):
    """
    分析权重相似性weight_sim，可视化
    :param doc_features: 攻击手法特征
    :param feature_dir: 特征保存目录
    :return:
    """
    weight_sim_pos, weight_sim_neg = [], []
    for doc_feature in doc_features:
        if 'weight_sim' in doc_feature['feature']:
            weight_sim_ = doc_feature['feature']['weight_sim']
            label = doc_feature['label']
            if label > 0:
                weight_sim_pos.append(weight_sim_[0])
            else:
                weight_sim_neg.append(weight_sim_[0])

    plot_and_save_histogram(weight_sim_pos, weight_sim_neg, feature_dir, "weight_sim")


def hash_sim(doc_features, feature_dir):
    """
    分析哈希相似性hash_sim，可视化
    :param doc_features: 攻击手法特征
    :param feature_dir: 特征保存目录
    :return:
    """
    # 分析keyword_match特征
    hash_sim_pos, hash_sim_neg = [], []
    for doc_feature in doc_features:
        if 'weight_sim' in doc_feature['feature']:
            hash_sim_ = doc_feature['feature']['hash_sim']
            label = doc_feature['label']
            if label > 0:
                hash_sim_pos.append(hash_sim_[0])
            else:
                hash_sim_neg.append(hash_sim_[0])

    plot_and_save_histogram(hash_sim_pos, hash_sim_neg, feature_dir, "hash_sim")


def get_keyword_and_doc_feature(data):
    """
    获取keyword_feature和doc_keyword
    :param data: 特征数据
    :return: 关键词特征、攻击手法特征
    """
    keyword_features = []
    doc_features = []
    for d in data:
        if 'feature' in d:
            feature = d['feature']
            label = d['label']
            if 'keyword_feature' in feature:
                keyword_features.append({
                    'label': label,
                    'feature': feature['keyword_feature']
                })
            if 'doc_feature' in feature:
                doc_features.append({
                    'label': label,
                    'feature': feature['doc_feature']
                })
    return keyword_features, doc_features


def analysis_feature(logger, feature_config, log_dir, filter_switch=False):
    """
    特征分析
    :param logger: 日志
    :param feature_config: 特征配置
    :param log_dir: 日志目录
    :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
    :return:
    """
    global glo_log_dir
    glo_log_dir = log_dir

    # 获取keyword_feature和doc_keyword
    feature_dir = feature_config['feature_dir']
    feature_filename = os.path.join(feature_dir, 'features.json')
    with open(feature_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keyword_features, doc_features = get_keyword_and_doc_feature(data)

    # 可视化各个特征
    keyword_match(keyword_features, feature_dir)
    keyword_dist_mean(keyword_features, feature_dir)
    keyword_dist_qt(logger, keyword_features, feature_dir)
    bayes_prob(keyword_features, feature_dir)
    if not filter_switch:
        doc_match(doc_features, feature_dir)
        weight_sim(doc_features, feature_dir)
        hash_sim(doc_features, feature_dir)

    logger.info("analysis feature done.")
