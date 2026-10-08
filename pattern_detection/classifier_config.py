"""
功 能：决策树分类器配置信息
"""

import os
import stat


class ClassifierConfig:
    """
    分类器配置信息
    """
    FLAGS = os.O_CREAT | os.O_RDWR
    MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP

    MODEL_PATH = "model/fast_fusion_model.pkl"

    CONTENT_LENGTH_THRESHOLD = 1000

    CLASSIFIER_THRESHOLD_NEG = 0.1

    CLASSIFIER_THRESHOLD_POS = 0.99

    # 判定结果无风险
    POSITIVE_FLAG = 0

    # 判定结果有风险
    NEGATIVE_FLAG = 1

    # 判定结果存疑
    DOUBT_FLAG = -1
