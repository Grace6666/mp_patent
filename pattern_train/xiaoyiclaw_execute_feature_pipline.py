#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 初始化配置，特征提取与分析pipline
"""

import os
import logging
import argparse

from sample_data import sample_data
from make_train_config import AttackPatternTrainConfig
from build_keyword_db import build_keyword_db
from extract_features import extract_features
from analysis_feature import analysis_feature
from extract_syntax_features import FeatureExtract


def init_env():
    """
    初始化环境变量
    """
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEVELOP_DATASET", type=str, help="xiaoyiclaw_feature_task_研发环境数据集")
    parser.add_argument("--PRODUCT_DATASET", type=str, help="xiaoyiclaw_feature_task_生产环境数据集")
    parser.add_argument("--OUTPUT_DIR", type=str, help="xiaoyiclaw_feature_task_结果输出路径")
    parser.add_argument("--LOG_DIR", type=str, help="xiaoyiclaw_feature_task_日志路径")
    parser.add_argument("--BASE_DIR", type=str, help="xiaoyiclaw_feature_task_工作目录")
    parser.add_argument("--FLAG", type=str, help="xiaoyiclaw_feature_task_时间版本")

    args, unknown = parser.parse_known_args()

    env_ = {
        "develop_dataset_dir": args.DEVELOP_DATASET,
        "product_dataset_dir": args.PRODUCT_DATASET,
        "output_dir": args.OUTPUT_DIR,
        "log_dir": args.LOG_DIR,
        "base_dir": args.BASE_DIR,
        "flag": args.FLAG,
    }

    return env_


if __name__ == "__main__":
    env = init_env()
    product_dataset_dir = env["product_dataset_dir"]
    develop_dataset_dir = env["develop_dataset_dir"]
    output_dir = env["output_dir"]
    log_dir = env["log_dir"]
    base_dir = env["base_dir"]
    flag = env["flag"]

    logging.basicConfig(
        filename=os.path.join(log_dir, 'xiaoyiclaw_execute_feature_pipline.log'),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()

    # 初始化配置
    logger.info("Start generate config!")
    data_dir = (develop_dataset_dir, product_dataset_dir)
    attack_pattern_train_config = AttackPatternTrainConfig(
        logger, base_dir=base_dir, flag=flag, data_dir=data_dir, output_dir=output_dir
    )
    attack_pattern_train_config.init_config()
    attack_pattern_train_config.update_data_config(flag)
    base_dir = attack_pattern_train_config.base_dir
    logger.info("Generate config done!")


    config = attack_pattern_train_config.config
    db_config = attack_pattern_train_config.config["db_config"]
    data_config = attack_pattern_train_config.config["data_config"]
    feature_config = attack_pattern_train_config.config["feature_config"]

    # 初始化句法特征提取器
    synax_features_extractor = FeatureExtract(logger, data_config)

    # 数据采样
    sample_data(data_config, logger, flag)

    # 构建关键词数据库
    build_keyword_db(logger, db_config, data_config, feature_config)

    # 特征提取
    extract_features(logger, config, synax_features_extractor)

    # 特征分析
    analysis_feature(logger, feature_config, log_dir, filter_switch=True)
