#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 训练数据构造
"""

import os
import json
import random

import numpy as np

from controller_feature import FastDetectorControllerFeature


def count_pos_neg_samples(controller, features, filter_switch=False):
    """
    统计正负样本数量
    :param controller: 特征提取控制器
    :param features: 特征配置
    :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
    :return: pos_num正样本数， neg_num负样本数
    """
    pos_num, neg_num = 0, 0
    for f in features:
        if 'feature' not in f:
            continue
        feature = f['feature']
        label = f['label']
        fuse_feature = controller.make_fuse_features(feature, filter_switch)
        if fuse_feature is None:
            continue
        neg_num, pos_num = update_counts(label, neg_num, pos_num)

    return pos_num, neg_num


def update_counts(label, neg_num, pos_num):
    """
    更新正负样本计数
    :param label: 样本标签
    :param pos_num: 当前正样本数
    :param neg_num: 当前负样本数
    :return: 更新后的pos_num正样本数， neg_num负样本数
    """
    if label > 0:
        pos_num += 1
    else:
        neg_num += 1
    return neg_num, pos_num


def calc_sample_rate(controller, features, fuse_config, filter_switch=False):
    """
    计算负样本采样率
    :param controller: 特征提取控制器
    :param features: 特征配置
    :param fuse_config: 融合fuse配置
    :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
    :return: neg_num负样本数， pos_num正样本数， sample_rate负样本采样率
    """
    pos_num, neg_num = count_pos_neg_samples(controller, features, filter_switch)

    neg_size = fuse_config['neg_size']
    if neg_num > pos_num * neg_size:
        sample_rate = (pos_num * neg_size) * 1.0 / neg_num
    else:
        sample_rate = 1.0

    return pos_num, neg_num, sample_rate


def build_training_data(controller, features, sample_rate, filter_switch=False):
    """
    构建训练数据
    :param controller: 特征提取控制器
    :param features: 特征配置
    :param sample_rate: 负样本采样率
    :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
    :return: neg_num负样本数，pos_num正样本数，fuse_features融合特征，labels标签
    """
    fuse_features, labels = [], []
    pos_num, neg_num = 0, 0
    for f in features:
        if 'feature' not in f:
            continue
        feature = f['feature']
        label = f['label']
        fuse_feature = controller.make_fuse_features(feature, filter_switch)
        if fuse_feature is None:
            continue
        # 检查是否应该包含此样本
        if not should_include_sample(label, sample_rate):
            continue
        fuse_features.append(fuse_feature)
        labels.append(label)
        neg_num, pos_num = update_counts(label, neg_num, pos_num)

    return pos_num, neg_num, fuse_features, labels


def should_include_sample(label, sample_rate):
    """
    判断是否应该包含该样本
    :param label: 样本标签
    :param sample_rate: 负样本采样率
    :return: 布尔值，是否包含该样本
    """
    # 正样本总是包含
    if label > 0:
        return True

    # 负样本根据采样率决定
    if label < 1 and random.random() < sample_rate:
        return True

    return False


def save_data(data_dir, xs, ys):
    """
    保存数据
    :param data_dir: 数据所在目录
    :param xs: 特征数据
    :param ys: 标签数据
    """
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, 'input.txt'), xs)
    np.savetxt(os.path.join(data_dir, 'label.txt'), ys, fmt='%d')


def make_rf_data(logger, config, filter_switch=False):
    """
    训练数据构造
    :param logger: 日志
    :param config: 配置
    :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
    :return:
    """
    fuse_config = config['fuse_config']
    feature_config = config['feature_config']
    data_dir = fuse_config['data_dir']
    feature_dir = feature_config['feature_dir']
    feature_filename = os.path.join(feature_dir, 'features.json')
    with open(feature_filename, 'r', encoding='utf-8') as f:
        features = json.load(f)

    controller = FastDetectorControllerFeature(config)
    pos_num, neg_num, sample_rate = calc_sample_rate(controller, features, fuse_config, filter_switch)
    logger.info(f"positive sample size: {pos_num}.    negative sample size: {neg_num}.    sample rate: {sample_rate}")
    if pos_num == 0 or neg_num == 0:
        logger.error("Sample number is abnormal. Failed to construct training data.")
        raise RuntimeError("Sample number is abnormal. Failed to construct training data.")

    pos_num, neg_num, fuse_features, labels = build_training_data(controller, features, sample_rate, filter_switch)
    logger.info(f"positive sample size: {pos_num}.    negative sample size: {neg_num}.    sample rate: {sample_rate}")

    xs = np.array(fuse_features)
    ys = np.array(labels, dtype=np.int32)
    save_data(data_dir, xs, ys)
    logger.info(f"fuse_data save at {data_dir}.")
