#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 关键词库
"""
import os
import json


# 关键词
class InjectionKeyword:
    # 关键词的ID
    idx: int = -1

    # 关键词内容
    word: str = ''

    # 关键词的权重
    weight: float = 0.0

    # 关键词的贝叶斯概率值
    bayes_prob: float = 0.0

    # 倒排索引
    inverse_docs: list = []

    def __init__(self):
        pass


# 攻击手法
class InjectionDoc:
    # 攻击手法的ID
    idx: int = -1

    # 攻击手法文本
    content: str = ''

    # 攻击手法经过分词之后的词列表
    words: list[str] = []

    # 攻击手法中每个词出现的次数
    word_counts: dict = {}

    # 攻击手法中每个关键词出现的次数
    keyword_counts: dict = {}

    # 攻击手法中每个词的权重，词典表示，用于计算weight_sim
    word_weights_map: dict = {}

    # 攻击手法的哈希值，用于计算hash_sim
    doc_hash: str = ''

    def __init__(self):
        pass


# 关键词库
class KeywordDB:
    docs: list[InjectionDoc] = []
    keywords: list[InjectionKeyword] = []

    # 给定字符串单词，反向搜索关键词的ID
    keywords_map: dict = {}

    # 给定字符串单词，返回单词的权重
    keywords_weight: dict = {}

    # 给定字符串单词，返回贝叶斯概率值
    keywords_bayes_prob: dict = {}

    # 配置文件
    config: dict = {
        'weight_threshold': 2
    }


    def __init__(self, config: dict = None):
        if config is not None:
            self.config = config

    def add_keywords(self, keyword_list):
        """
        权重超过阈值的关键词添加关键词库
        :param keyword_list: [word_id, word, weight, inverse_doc_ids]
        :return: None
        """
        word_id = 0
        for _, word, weight, doc_ids in keyword_list:
            if weight > self.config['weight_threshold']:
                keyword = InjectionKeyword()
                keyword.idx = word_id
                keyword.word = word
                keyword.weight = weight
                keyword.inverse_docs = doc_ids
                self.keywords.append(keyword)
                self.keywords_map[word] = word_id
                self.keywords_weight[word] = weight
                word_id += 1

    def add_docs(self, doc_list):
        """
        攻击手法入库，统计攻击手法中关键词、词出现的次数、权重等信息
        :param doc_list: [doc_id, content, words]
        :return:
        """
        for doc_id, ctx, words in doc_list:
            doc = InjectionDoc()
            doc.idx = doc_id
            doc.content = ctx
            doc.words = words
            doc.word_counts = {}
            doc.keyword_counts = {}
            doc.word_weights_map = {}

            self.calculate_word_weights(doc, words)

            self.docs.append(doc)

    def calculate_word_weights(self, doc, words):
        """
        计算文档中词的权重，并统计关键词和词的出现次数
        :param doc: 文档对象，包含关键词计数、词计数和词权重映射属性。
        :param words: 文档中的词列表。
        :return: 无返回值，但会更新传入的doc对象的keyword_counts、word_counts和word_weights_map属性。
        """
        for word in words:
            # word是关键词，加入关键词计数
            if word in self.keywords_map:
                if word not in doc.keyword_counts:
                    doc.keyword_counts[word] = 0
                doc.keyword_counts[word] += 1

            # word计数
            if word not in doc.word_counts:
                doc.word_counts[word] = 0
            doc.word_counts[word] += 1

            # 统计文档中所有词的权重
            if word not in doc.word_weights_map:
                if word in self.keywords_weight:
                    doc.word_weights_map[word] = self.keywords_weight[word]
                else:
                    doc.word_weights_map[word] = 1

    def fetch_docs_words(self, doc_ids: list[int]):
        """
        给定文档ID集合，顺序返回文档分词后的序列
        :param doc_ids: 需要查询的doc_id列表
        :return: 顺序返回各个doc的分词后列表
        """
        return [self.docs[idx].words for idx in doc_ids]

    def fetch_all_docs_words(self):
        """
        返回所有文档分词后的序列
        :return: 顺序返回文档分词后列表
        """
        return [(idx, self.docs[idx].words) for idx in range(len(self.docs))]

    def fetch_inverse_doc_ids(self, keyword_ids: list[int]):
        """
        给定关键词，获取每个关键词对应的倒排索引文档集合
        :param keyword_ids: 需要查询的关键词列表
        :return: [inverse_ids]
        """
        return [self.keywords[idx].inverse_docs for idx in keyword_ids]

    def fetch_all_keywords(self):
        """
        获取所有关键词字符串，返回词列表
        :return: [keyword]
        """
        results = [self.keywords[idx].word for idx in range(len(self.keywords))]
        return results

    def fetch_docs_for_weight_sim(self, doc_ids: list[int]):
        """
        提供weight sim算法所需的数据，即指定文档的相关信息，包括文档分词后的序列，以及文档中各个词的权重
        :param doc_ids: 需要查询的doc_id列表
        :return: [words, word_weights_map]，后者是文档中每个词的权重（不是关键词）
        """
        return [(self.docs[idx].words, self.docs[idx].word_weights_map) for idx in doc_ids]

    def fetch_docs_for_hash_sim(self, doc_ids: list[int]):
        """
        提供hash sim算法所需的数据，即文档的哈希值
        :param doc_ids: 需要查询的doc_id列表
        :return: [doc_hash]
        """
        return [self.docs[idx].doc_hash for idx in doc_ids]

    def fetch_page_keyword_weights(self, words: list[str]):
        """
        hash_sim计算使用的数据，返回页面的关键词权重
        :param words: 页面数据分词后的结果
        :return: 页面每个词对应的权重列表，非关键词用0填充，返回的长度跟原界面一样
        """
        m = len(words)
        weights = [0] * m
        # 遍历每个word 如果word在关键词权重库中，记录其权重，并返回
        for i in range(m):
            word = words[i]
            if word in self.keywords_weight:
                weights[i] = self.keywords_weight[word]
        return weights

    def set_docs_hash(self, docs_hash: list[(int, str)]):
        """
        设置文档库所有文档的哈希值，输入为的序列
        :param docs_hash: [doc_id, hash]
        :return: None
        """
        for idx, doc_hash in docs_hash:
            self.docs[idx].doc_hash = doc_hash

    def fetch_page_keyword_bayes_probs(self, words: list[str]):
        """
        bayes分类器计算使用的数据，返回页面关键词的贝叶斯概率值
        :param words: 页面分词后的词列表
        :return: 权重列表，如果当前词是关键词才添加在权重列表中，不用0填充，长度可能小于原界面
        """
        m = len(words)
        weights = []
        for i in range(m):
            word = words[i]
            if word in self.keywords_bayes_prob:
                weights.append(self.keywords_bayes_prob[word])
        return weights

    def fetch_page_keyword_bayes_probs_ngram(self, words: list[str]):
        """
        基于n-gram的贝叶斯分类器所需的数据，返回贝叶斯概率值以及下标
        :param words: 页面分词后的词列表  匹配的关键词
        :return: 权重列表，格式为(id, weight)，id指的是在words中第几个
        """
        m = len(words)
        weights = []
        for i in range(m):
            word = words[i]
            if word in self.keywords_bayes_prob:
                weights.append((i, self.keywords_bayes_prob[word], word))
        return weights

    def load_from_file(self, file_dir: str):
        """
        加载关键词数据，包括关键词、攻击模式 到 KeywordDB
        :param file_dir: 关键词数据库文件目录
        """
        # 关键词
        keyword_list = []
        with open(os.path.join(file_dir, 'keywords.json'), 'r', encoding='utf-8') as f:
            data_keywords = json.load(f)
            for data in data_keywords:
                keyword_list.append((data['idx'], data['word'], data['weight'], data['inverse_docs']))
        self.add_keywords(keyword_list)

        # 攻击模式
        doc_list = []
        with open(os.path.join(file_dir, 'docs.json'), 'r', encoding='utf-8') as f:
            data_docs = json.load(f)
            for data in data_docs:
                doc_list.append((data['idx'], data['context'], data['words']))
        self.add_docs(doc_list)

    # 从文件目录中加载贝叶斯分类器相关信息，需先完成数据库构建存储，以及针对训练页面数据的特征分析后才能调用此函数
    def load_bayes_prob(self, file_dir: str):
        """
        从文件目录中加载贝叶斯分类器相关信息，需先完成数据库构建存储，以及针对训练页面数据的特征分析后才能调用此函数
        :param file_dir: 文件目录字符串，绝对路径
        """
        with open(os.path.join(file_dir, 'bayes_prob.json'), 'r', encoding='utf-8') as f:
            # bayes_probs是读取到的
            bayes_probs = json.load(f)
            for word in bayes_probs:
                if word in self.keywords_map:
                    # keywords_map倒排索引，获得的是在keywords中的索引，此处的keywords都是筛选过后超过阈值的关键词
                    idx = self.keywords_map[word]
                    self.keywords[idx].bayes_prob = bayes_probs[word]
                    # 字符串倒排bayes概率索引
                    self.keywords_bayes_prob[word] = bayes_probs[word]
