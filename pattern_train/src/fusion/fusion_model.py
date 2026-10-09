#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 融合评估模型
修改时间: 2025/8/1 10:00
"""

import os
import pickle

import numpy as np


class FuseModel:

    # 模型超参数
    config: dict = {
        'top_docs': 1
    }

    # 融合评估模型
    model = None

    def __init__(self, config: dict = None):
        if config is not None:
            self.config = config

    def fuse_features(self, key_feature: dict, doc_feature: dict):
        """
        融合特征，把关键词和文档特征拼接成一个长向量
        :param key_feature: 关键词特征，字典表示
        :param doc_feature: 文档特征，字典表示
        :return: 特征向量列表
        """
        features = []
        features.append(key_feature['keyword_match'])
        key_dist_feature = key_feature['dist_feature']
        features.append(key_dist_feature['mean'])
        features.extend(key_dist_feature['qt'])
        features.extend(doc_feature['hash_sim'][0:self.config['top_docs']])
        features.extend(doc_feature['weight_sim'][0:self.config['top_docs']])
        return features

    @staticmethod
    def fuse_seg_and_key_features(seg_feature: list, key_feature: dict):
        """
        融合句法特征及关键词特征，拼成一个长向量
        :param seg_feature: 句法特征
        :param key_feature: 关键词特征，字典表示
        :return: 特征向量列表
        """
        key_dist_feature = key_feature['dist_feature']
        features = seg_feature + [key_feature['keyword_match'], key_dist_feature['mean']]
        features.extend(key_dist_feature['qt'])
        return features

    def judge_by_fuse_features(self, fuse_feature: list):
        """
        通过模型判断是否放行
        :param fuse_feature: 融合拼接后的特征向量
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        x = np.expand_dims(np.array(fuse_feature), axis=0)
        pred = self.model.predict_proba(x)[0, 1]
        if pred < self.config['classifier']['threshold']['neg']:
            return -1
        if pred > self.config['classifier']['threshold']['pos']:
            return 1
        return 0
    
    def debug_by_fuse_features(self, fuse_feature: list):
        """
        通过模型判断是否放行
        :param fuse_feature: 融合拼接后的特征向量
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        x = np.expand_dims(np.array(fuse_feature), axis=0)
        pred = self.model.predict_proba(x)[0, 1]
        if pred < self.config['classifier']['threshold']['neg']:
            return -1, pred
        if pred > self.config['classifier']['threshold']['pos']:
            return 1, pred
        return 0, pred

    def load_model_from_file(self, model_dir):
        """
        加载模型，需先完成特征构建、模型训练，才能调用此函数
        :param model_dir: 模型目录，绝对路径
        """
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
