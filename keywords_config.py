#!/usr/bin/env python
# coding: utf-8
"""
功 能：关键词特征相关配置参数
"""

import os
import stat


class FilePermissions:
    """
    文件权限设置
    """
    FLAGS = os.O_CREAT | os.O_RDWR
    MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP


class KeywordsBaseConfig:
    """
    关键词库配置信息
    """
    KEYWORDS_BASE_PATH = "knowledge_base/keywords.json"

    # 停用词库路径
    STOP_WORDS_BASE_PATH = "knowledge_base/stopwords.txt"

    # 关键词贝叶斯概率文件路径
    KEYWORDS_BAYES_PROB_PATH = "knowledge_base/bayes_prob.json"


class KeywordsFeatureIsDetected:
    """
    关键词特征是否检出攻击配置
    """
    KEYWORD_COUNT_IS_DETECTED = False

    KEYWORD_BAYES_PROB_IS_DETECTED = True

    KEYWORD_DIST_IS_DETECTED = False


class KeywordsFeatureConfig:
    """
    关键词特征相关配置参数
    """

    KEYWORD_MATCH_THRESHOLD = {"neg": 2, "pos": 100}

    BAYES_PROB_THRESHOLD = {"neg": 0.0, "pos": 5.0}

    BAYES_NGRAM = 3

    BAYES_NGRAM_MARGIN = 3

    WEIGHT_THRESHOLD = 1

    QT_POS_RATE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    QT_CHECK_POS = [0]

    QT_CHECK_THRESHOLD = [15]

    # 判定结果无风险
    POSITIVE_FLAG = 0

    # 判定结果有风险
    NEGATIVE_FLAG = 1

    # 判定结果存疑
    DOUBT_FLAG = -1


