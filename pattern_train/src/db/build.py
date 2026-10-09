#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 构建数据库工具
"""

from utils import cut_words, judge_end_with_punc


def calc_keyword_counts_threshold(keywords_, keyword_counts):
    """
    计算得到权重对应出现次数阈值列表
    :param keywords_: 关键词信息，关注出现次数
    :param keyword_counts: 出现次数比例阈值
    :return:
    """
    # 所有关键词出现次数
    all_keyword_counts = []
    for keyword in keywords_.keys():
        count = keywords_[keyword]['count']
        all_keyword_counts.append(count)
    all_keyword_counts_ = sorted(all_keyword_counts, reverse=True)

    # 权重对应出现次数阈值
    keyword_counts_threshold = []
    for c in keyword_counts:
        c_new_idx = int(len(all_keyword_counts_) * c)
        keyword_counts_threshold.append(all_keyword_counts_[c_new_idx])

    return keyword_counts_threshold


def calc_weight(count, keyword_counts_threshold, keyword_weights):
    """
    计算该关键词的权重
    :param count: 关键词出现次数
    :param keyword_counts_threshold: 权重对应出现次数阈值
    :param keyword_weights: 权重列表
    :return: 关键词的权重
    """
    # 根据出现次数计算权重
    for keyword_count, keyword_weight in zip(keyword_counts_threshold, keyword_weights):
        if count > keyword_count:
            return keyword_weight
    return 1


def process_words(words, stop_words, keywords_, doc_idx):
    """
    处理words，得到keywords_
    :param words: 待处理的单词列表
    :param stop_words: 停顿词
    :param keywords_: 关键词字典
    :param doc_idx: 文档索引
    :return: 更新后的关键词字典
    """
    for word in words:
        word = word.strip()
        if len(word) > 0 and word not in stop_words and not judge_end_with_punc(word):
            if word not in keywords_:
                keywords_[word] = {}
                keywords_[word]['count'] = 0
                keywords_[word]['inverse'] = set()
            keywords_[word]['count'] += 1
            keywords_[word]['inverse'].add(doc_idx)
    return keywords_


def build_docs(hijack_ctx:dict, stop_words:set):
    """
    构建doc_list列表，写入doc.json
    :param hijack_ctx: 攻击手法
    :param stop_words: 停顿词
    :return:
    """
    keywords_ = {}
    doc_list = []
    doc_set = set()
    doc_idx = 0

    # 构造doc_list
    for _, ctx in hijack_ctx:
        if ctx in doc_set:
            continue
        doc_set.add(ctx)
        words = cut_words(ctx)
        doc_list.append(
            {'idx': doc_idx,
             'context': ctx,
             'words': words
             }
        )
        keywords_ = process_words(words, stop_words, keywords_, doc_idx)

        doc_idx += 1

    return doc_list, keywords_


def build_keywords(keywords_:dict, keyword_counts: list[int], keyword_weights: list[int]):
    """
    构建keyword_list列表，写入keywords.json
    :param keywords_: 关键词信息
    :param keyword_counts: 关键词出现次数
    :param keyword_weights: 关键词出现次数对应的权重
    :return: 构造的keyword_list
    """
    keyword_list = []
    word_idx = 0
    # 计算出现权重对应次数阈值
    keyword_counts_threshold = calc_keyword_counts_threshold(keywords_, keyword_counts)
    # 构造keyword_list
    for keyword in keywords_.keys():
        count, inverse = keywords_[keyword]['count'], list(keywords_[keyword]['inverse'])
        weight = calc_weight(count, keyword_counts_threshold, keyword_weights)
        keyword_list.append(
            {
                'idx': word_idx,
                'word': keyword,
                'weight': weight,
                'inverse_docs': inverse
            }
        )
        word_idx += 1

    return keyword_list


def build_keywords_and_docs(hijack_ctx:dict, keyword_counts: list[int], keyword_weights: list[int], stop_words: set):
    """
    从攻击模式库中提取出keyword和doc列表，写入json文件，用于KeywordDB的初始化
    :param hijack_ctx: 攻击手法文档集合json文件
    :param keyword_counts: 关键词出现次数阈值列表
    :param keyword_weights: 关键词权重列表
    :param stop_words: 停顿词集合
    :return: 返回doc和keyword列表
    """
    doc_list, keywords_ = build_docs(hijack_ctx, stop_words)
    keyword_list = build_keywords(keywords_, keyword_counts, keyword_weights)

    return keyword_list, doc_list
