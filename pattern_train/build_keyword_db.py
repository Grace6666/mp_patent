#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 构建关键词库
"""

import json
import os
import math
import pickle

from keywords.ac import AC
from db.keyword_db import KeywordDB
from utils import remove_stop_words
from db.build import build_keywords_and_docs
from keywords.keyword_model import KeywordModel


def load_attack_modes(attack_modes_filename:str = None):
    """
    读取攻击模式
    :param attack_modes_filename: 攻击模式库
    :return: 攻击模式hijack_ctx
    """
    # 读取攻击手法
    with open(attack_modes_filename, 'rb') as f:
        hijack_ctx = pickle.load(f)
    return hijack_ctx


def load_stopwords(stopword_filename:str = None):
    """
    读取停顿词
    :param stopword_filename: 停顿词文件名
    :return: 停顿词stop_words
    """
    # 读取停顿词信息
    stop_words = set()
    with open(stopword_filename, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    return stop_words


def load_pages(pages_filename):
    """
    读取页面信息
    :param pages_filename: 页面保存位置
    :return: None
    """
    with open(pages_filename, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    return pages


def write_keywords_and_docs_db(hijack_ctx, stop_words, keyword_counts, keyword_weights, keyword_db_dir):
    """
    从攻击模式库中提取出keywords和docs列表，写入json文件中，后续用于KeywordDB的初始化
    :param hijack_ctx: dict 攻击模式库
    :param stop_words: set 停顿词
    :param keyword_counts: 关键出现次数
    :param keyword_weights: 关键词权重，与关键词出现次数对应
    :param keyword_db_dir: 解析出的关键词数据库的存放位置
    :return: None
    """
    os.makedirs(keyword_db_dir, exist_ok=True)

    keyword_list, doc_list = build_keywords_and_docs(hijack_ctx, keyword_counts, keyword_weights, stop_words)
    keyword_list_filename = os.path.join(keyword_db_dir, 'keywords.json')
    doc_list_filename = os.path.join(keyword_db_dir, 'docs.json')

    with open(keyword_list_filename, 'w', encoding='utf-8') as f:
        json.dump(keyword_list, f, ensure_ascii=False)
    with open(doc_list_filename, 'w', encoding='utf-8') as f:
        json.dump(doc_list, f, ensure_ascii=False)


def cal_and_write_bayes_prob(keywords, keyword_feature_config, pages, stop_words, prob_filename):
    """
    根据关键词库与页面库计算bayes_prob
    :param keywords: 关键词库
    :param keyword_feature_config:关键词特征配置
    :param pages: 页面库
    :param stop_words: 停顿词集合
    :param prob_filename: bayes_prob.json文件
    :return: None
    """
    # 初始化关键词特征模型
    keyword_model = KeywordModel(ac=AC(keywords), config=keyword_feature_config)

    # 计算贝叶斯概率前处理
    probs = {}
    for i in range(len(pages)):
        data = pages[i]
        label = data['label']
        page = data['page']
        keyword_feature = {}

        # 页面关键词匹配并根据关键词匹配数量判定是否放行
        _, page = remove_stop_words(page, stop_words)
        matched_keywords_pos, matched_keywords_idx, matched_keywords = keyword_model.keyword_match(page)
        keyword_feature['keyword_match'] = len(matched_keywords)
        if keyword_model.judge_by_keyword_match(keyword_feature) < 0:
            continue
        for keyword in matched_keywords:
            if keyword not in probs:
                probs[keyword] = [0.0, 0.0]
            probs[keyword][label] += 1

    # 计算bayes_probs并写入bayes_prob.json
    for keyword in probs:
        probs[keyword].append(
            (math.exp(probs[keyword][1] / (probs[keyword][0] + probs[keyword][1])) - 1) / (math.exp(1) - 1))
    bayes_prob = {}
    for keyword in probs:
        bayes_prob[keyword] = probs[keyword][2]
    with open(prob_filename, 'w', encoding='utf-8') as f:
        json.dump(bayes_prob, f, ensure_ascii=False)


def build_keyword_db(logger, db_config:dict = None, data_config:dict = None, feature_config:dict = None):
    """
    构建关键词数据库
    :param logger:日志对象
    :param db_config: 关键词数据库配置
    :param data_config: 训练数据配置
    :param feature_config:
    :return:
    """
    # 读取停顿词、攻击手法
    hijack_ctx = load_attack_modes(data_config['attack_modes_filename'])
    stop_words = load_stopwords(data_config['stopwords_filename'])
    logger.info("Read stopwords and attack_modes done!")

    # 解析攻击模式，写入keyword_db数据库
    keyword_counts = db_config['keyword']['counts']
    keyword_weights = db_config['keyword']['weights']
    keyword_db_dir = db_config['keyword_dir']
    write_keywords_and_docs_db(hijack_ctx, stop_words, keyword_counts, keyword_weights, keyword_db_dir)
    logger.info("Build keywords_db and docs_db done!")

    # 加载训练数据页面
    page_filename = os.path.join(data_config['page_dir'], 'save_train_pages.json')
    pages = load_pages(page_filename)
    logger.info("Load train pages data done!")

    # 定义keywordDB类
    keyword_db = KeywordDB(db_config['keyword'])
    keyword_db.load_from_file(keyword_db_dir)
    keywords = keyword_db.fetch_all_keywords()

    # 计算bayes_prob
    prob_filename = os.path.join(db_config['keyword_dir'], 'bayes_prob.json')
    keyword_feature_config = feature_config['keyword']
    cal_and_write_bayes_prob(keywords, keyword_feature_config, pages, stop_words, prob_filename)
