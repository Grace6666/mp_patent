#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 计算词与句子的哈希值
"""

import numpy as np


def string_hash(s: str):
    """
    计算单个词的哈希值
    :param s: 字符串
    :return: 哈希值
    """
    if s == '':
        return '0'

    x = ord(s[0]) << 7
    m = 1000003
    mask = 2 ** 128 - 1

    for c in s:
        x = ((x * m) ^ ord(c)) & mask
    x ^= len(s)

    if x == -1:
        x = -2

    x = bin(x).replace('0b', '').zfill(64)[-64:]
    return str(x)


def sentence_hash(segs: list[str], weights: list[int]):
    """
    计算整个句子的哈希值
    :param segs: 句子中的每个单词
    :param weights: 每个单词对应的权重
    :return: 句子哈希值
    """
    scores = []
    n = len(segs)

    for i in range(n):
        seg = segs[i]
        w = weights[i]
        s = string_hash(seg)

        score = []
        for c in s:
            weight = 1 if c == '1' else 0
            weight *= w
            score.append(weight)
        scores.append(score)

    scores = np.array(scores).sum(0)
    result = ''
    for score in scores:
        if score > 0:
            result += '1'
        else:
            result += '0'

    return result


def calc_distance_between_hash(h1: str, h2: str):
    """
    计算两个哈希值之间的距离，0-1表示，越接近1相似度越高
    :param h1: 哈希值1
    :param h2: 哈希值2
    :return: 相似度
    """
    n = min(len(h1), len(h2))
    count = 0
    for i in range(n):
        if h1[i] == h2[i]:
            count += 1
    return count * 1.0 / n


def calc_weights_(segs, keyword_weight):
    """
    测试用接口
    :param segs:
    :param keyword_weight:
    :return:
    """
    results = []
    for seg in segs:
        if seg in keyword_weight:
            results.append(keyword_weight[seg])
        else:
            results.append(0)
    return results
