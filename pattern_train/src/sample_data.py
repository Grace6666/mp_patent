#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 快筛模块训练数据采样
"""

import os
import json
import random


def sample_data(data_config, logger, flag):
    """
    数据采样
    """
    source_data_path = data_config["source_data_path"]
    sample_num = data_config["sample_num"]
    output_dir = os.path.join(data_config["page_dir"], flag)
    data_config["page_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始正负样本
    try:
        with open(source_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Load source_data_path {source_data_path} failed: {e}")
        return

    # 按照sample_num进行随机采样
    if isinstance(data, list):
        selected_samples = random.sample(data, min(sample_num, len(data)))
    else:
        logger.warning(f"The data in {source_data_path} is not in the list format, and random sampling failed.")
        selected_samples = []

    # 写入
    output_file = os.path.realpath(os.path.join(output_dir, "save_train_pages.json"))
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_samples, f, ensure_ascii=False, indent=4)
        logger.info(f"Write {len(selected_samples)} samples to {output_file}.")
    except Exception as e:
        logger.error(f"Failed to write to {output_file}: {e}")
