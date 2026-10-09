#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 训练参数读取与更新
修改时间: 2025/8/26 10:00
"""
import os
import json


def load_config(filename):
    """
    加载config配置文件
    :param filename: json配置文件的绝对路径
    :return: 配置字典
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def load_controller_config(filename):
    """
    加载controller配置文件
    :param filename: controller配置文件的绝对路径
    :return: 配置字典
    """
    with open(filename, 'r', encoding='utf-8') as f:
        controller_config = json.load(f)

    config_dir = controller_config['config_dir']
    config_name = ['db_config', 'data_config', 'fuse_config', 'feature_config']
    config_filename = {name: os.path.join(config_dir, controller_config[name]) for name in config_name}
    config = {key: load_config(path) for key, path in config_filename.items()}
    config['config_dir'] = config_dir

    return config


def read_train_config(base_dir):
    """
    :param base_dir: train.config所在目录
    """
    filename = os.path.join(base_dir, "train.config")
    with open(filename, encoding="utf-8") as config_file:
        model_config = json.load(config_file)
    return model_config


def upgrade_model_config(base_dir, config_dir, flag):
    """
    :param base_dir: train.config所在目录
    :param config_dir: 配置文件所在目录
    :param flag: 版本标志
    """
    new_model_config = read_train_config(base_dir)
    features_config_filename = os.path.join(config_dir, f"feature_{flag}.json")
    with open(features_config_filename, 'r', encoding="utf-8") as file:
        features_config = json.load(file)

    # 修改参数
    keywords_config = features_config["keyword"]
    new_keywords_config = new_model_config["keywords_config"]
    keywords_config["qt_pos_rate"] = new_keywords_config["QT_POS_RATE"]
    keywords_config["qt_check_pos"] = new_keywords_config["QT_CHECK_POS"]
    keywords_config["qt_check_threshold"] = new_keywords_config["QT_CHECK_THRESHOLD"]
    keywords_config["keyword_match_threshold"]["neg"] = new_keywords_config["KEYWORD_MATCH_THRESHOLD_NEG"]
    keywords_config["keyword_match_threshold"]["pos"] = new_keywords_config["KEYWORD_MATCH_THRESHOLD_POS"]
    keywords_config["bayes_prob_threshold"]["neg"] = new_keywords_config["BAYES_PROB_THRESHOLD_NEG"]
    keywords_config["bayes_prob_threshold"]["pos"] = new_keywords_config["BAYES_PROB_THRESHOLD_POS"]
    keywords_config["bayes_ngram"] = new_keywords_config["BAYES_NGRAM"]
    keywords_config["bayes_ngram_margin"] = new_keywords_config["BAYES_NGRAM_MARGIN"]
    keywords_config["weight_threshold"] = new_keywords_config["WEIGHT_THRESHOLD"]
    features_config["keyword"] = keywords_config

    doc_config = features_config["doc"]
    new_doc_config = new_model_config["paragraph_config"]
    doc_config["jaccard_threshold"] = new_doc_config["JACCARD_THRESHOLD"]
    doc_config["top_inverse_docs"] = new_doc_config["TOP_INVERSE_DOCS"]
    doc_config["hash_sim_threshold"]["neg"] = new_doc_config["HASH_SIM_THRESHOLD_NEG"]
    doc_config["hash_sim_threshold"]["pos"] = new_doc_config["HASH_SIM_THRESHOLD_POS"]
    doc_config["weight_sim_threshold"]["neg"] = new_doc_config["WEIGHT_SIM_THRESHOLD_NEG"]
    doc_config["weight_sim_threshold"]["pos"] = new_doc_config["WEIGHT_SIM_THRESHOLD_POS"]
    doc_config["window_size"] = new_doc_config["WINDOW_SIZE"]
    features_config["doc"] = doc_config

    fuse_config = features_config["fuse"]
    new_fuse_config = new_model_config["classifier_config"]
    fuse_config["top_docs"] = new_doc_config["TOP_DOCS"]
    fuse_config["classifier"]["threshold"]["neg"] = new_fuse_config["CLASSIFIER_THRESHOLD_NEG"]
    fuse_config["classifier"]["threshold"]["pos"] = new_fuse_config["CLASSIFIER_THRESHOLD_POS"]
    features_config["fuse"] = fuse_config

    with open(features_config_filename, "w", encoding="utf-8") as file:
        json.dump(features_config, file, ensure_ascii=False, indent=4)
