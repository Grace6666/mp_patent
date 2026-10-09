#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 攻击手法模型
"""

from doc.weight_sim import calc_weight_sim
from doc.hash_sim import sentence_hash, calc_distance_between_hash


class DocModel:
    config: dict = {
        'jaccard_threshold': 5,
        'top_inverse_docs': 3,
        'window_size': 256,
        'text_sim': ['weight', 'hash'],
        'hash_sim_threshold': 0.6,
        'weight_sim_threshold': 0.1
    }

    def __init__(self, config=None):
        if config is not None:
            self.config = config

    def jaccard_filter(self, inverse_ids):
        """
        给定倒排索引匹配到的文档，输出过滤之后的文档集合
        过滤思路：计算page跟各个文档在关键词上的jaccard系数，大于某个阈值则筛选出来
        计算方法：基于AC自动机匹配结果，只需要枚举匹配到的Keyword，看倒排索引中doc_id出现次数即可
        :param inverse_ids: 每个匹配到关键词的倒排索引文档，两层list
        :return: 满足jaccard系数阈值的文档列表
        """
        matched_docs = {}
        for docs_id in inverse_ids:
            for doc_id in docs_id:
                if doc_id not in matched_docs:
                    matched_docs[doc_id] = 0
                matched_docs[doc_id] += 1

        filtered_docs_id = []
        for doc_id in matched_docs:
            if matched_docs[doc_id] > self.config['jaccard_threshold']:
                filtered_docs_id.append((doc_id, matched_docs[doc_id]))

        filtered_docs_id = sorted(filtered_docs_id, key=(lambda x: x[1]), reverse=True)
        results = []
        n = min(self.config['top_inverse_docs'], len(filtered_docs_id))
        for i in range(n):
            results.append(filtered_docs_id[i][0])

        return results

    def calc_hash_sims(self, page_words, page_weights, docs_for_hash_sim):
        """
        计算页面跟文档之间的哈希相似度
        :param page_words: 页面所有词的列表
        :param page_weights: 页面中所有词的权重，如果不是关键词或者关键词的阈值过低则置0
        :param docs_for_hash_sim: 相似攻击手法的哈希值
        :return: 相似度从高到低排序，页面跟文档的哈希值列表
        """
        results = []
        page_hash = sentence_hash(page_words, page_weights)
        for doc_hash in docs_for_hash_sim:
            results.append(calc_distance_between_hash(page_hash, doc_hash))
        return sorted(results, reverse=True)[0:self.config['top_inverse_docs']]

    def calc_weight_sims(self, page_words, docs_for_weight_sim):
        """
        计算页面跟攻击手法文档之间的词权重相似度
        :param page_words: 页面所有词的列表
        :param docs_for_weight_sim: 文档权重信息列表，每个文档都由(文档所有的词，文档所有词对应权重）的列表组成
        :return: 相似度从高到底排序，页面跟文档的权重值列表
        """
        results = []
        for doc_words, doc_word_weight_map in docs_for_weight_sim:
            results.append(calc_weight_sim(page_words, doc_words, doc_word_weight_map))
        return sorted(results, reverse=True)[0:self.config['top_inverse_docs']]

    def judge_by_inverse_docs(self, doc_features):
        """
        通过检索到文档的数量是否小于阈值
        :param doc_features: 文档特征字典，包含top_inverse_docs字段
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if 'doc_match' not in doc_features:
            return 0

        if doc_features['doc_match'] < self.config['top_inverse_docs']:
            return -1
        return 0

    def judge_by_hash_sim(self, doc_features):
        """
        通过哈希值判断是否放行
        :param doc_features: 文档特征字典，包含hash_sim字段
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if 'hash_sim' not in doc_features:
            return 0

        hash_sims = doc_features['hash_sim']
        if hash_sims[0] < self.config['hash_sim_threshold']['neg']:
            return -1
        if hash_sims[0] > self.config['hash_sim_threshold']['pos']:
            return 1
        return 0

    def judge_by_weight_sims(self, doc_features):
        """
        通过权重相似度判断是否放行
        :param doc_features: 文档特征字典，包含weight_sim字段
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if 'weight_sim' not in doc_features:
            return 0

        weight_sims = doc_features['weight_sim']
        if weight_sims[0] < self.config['weight_sim_threshold']['neg']:
            return -1
        if weight_sims[0] > self.config['weight_sim_threshold']['pos']:
            return 1
        return 0
