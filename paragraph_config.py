#!/usr/bin/env python
# coding: utf-8
"""
功 能：文本特征相关配置参数
"""


class PatternBaseConfig:
    """
    攻击模式库相关配置
    """
    # 攻击模式库路径
    PATTERN_BASE_PATH = "knowledge_base/docs.json"


class ParagraphFeatureIsDetected:
    """
    文本特征是否检出攻击配置
    """
    MATCHED_PATTERN_IS_DETECTED = False

    WEIGHT_SIMILARITY_IS_DETECTED = False

    HASH_SIMILARITY_IS_DETECTED = False


class ParagraphFeatureConfig:
    """
    文本特征相关配置参数
    """
    JACCARD_THRESHOLD = 3

    TOP_INVERSE_DOCS = 3

    # 文本匹配正则
    SPLIT_TEMPLATE = r"[\u2000-\u206f\u3000-\u303f\uff00-\uffef]"

    WINDOW_SIZE = 256

    TEXT_SIM = ["weight", "hash"]

    HASH_SIM_THRESHOLD = {"neg": 0.8, "pos": 1.0}

    WEIGHT_SIM_THRESHOLD = {"neg": 0.1, "pos": 10.0}

    TOP_DOCS = 1

    # 判定结果无风险
    POSITIVE_FLAG = 0

    # 判定结果有风险
    NEGATIVE_FLAG = 1

    # 判定结果存疑
    DOUBT_FLAG = -1
