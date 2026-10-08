#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 小艺claw注入攻击检测快筛决策树模型训练
"""

import os
import logging
import argparse

from train_rf import train_rf
from make_rf_data import make_rf_data
from model_config.model_config import upgrade_model_config, load_controller_config


def init_env():
    """
    初始化环境变量
    """
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEVELOP_DATASET", type=str, help="xiaoyiclaw_model_train_研发环境数据集")
    parser.add_argument("--PRODUCT_DATASET", type=str, help="xiaoyiclaw_model_train_生产环境数据集")
    parser.add_argument("--OUTPUT_DIR", type=str, help="xiaoyiclaw_model_train_结果输出路径")
    parser.add_argument("--LOG_DIR", type=str, help="xiaoyiclaw_model_train_日志路径")
    parser.add_argument("--BASE_DIR", type=str, help="xiaoyiclaw_model_train_工作目录")
    parser.add_argument("--FLAG", type=str, help="xiaoyiclaw_model_train_时间版本")

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
        filename=os.path.join(log_dir, "model_train.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()

    # 训练配置更新
    config_dir = os.path.join(product_dataset_dir, "config", flag)
    logger.info("start upgrade model config!")
    upgrade_model_config(base_dir, config_dir, flag)
    logger.info("end upgrade model config!")

    # 新训练配置读取
    logger.info("start read new train config!")
    controller_config_filename = os.path.join(config_dir, f'controller_{flag}.json')
    config = load_controller_config(controller_config_filename)
    fuse_config = config['fuse_config']
    logger.info("end read new train config!")

    # 训练数据构造
    make_rf_data(logger, config, filter_switch=True)

    # 模型训练
    train_rf(logger, fuse_config, base_dir)
