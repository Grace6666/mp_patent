#!/usr/bin/env python
# coding: utf-8
"""
功 能：计算文本哈希相似度
"""

import numpy as np


class HashSimilarity:
    """
    文本哈希相似度计算类
    """
    @staticmethod
    def _string_hash(seq):
        """
        计算单个词的哈希值
        :param seq: 字符串
        :return: 哈希值
        """
        if seq == "":
            return "0"

        x = ord(seq[0]) << 7
        m = 1000003
        mask = 2 ** 128 - 1

        for char in seq:
            x = ((x * m) ^ ord(char)) & mask
        x ^= len(seq)

        if x == -1:
            x = -2

        x = bin(x).replace("0b", "").zfill(64)[-64:]
        return str(x)

    def sentence_hash(self, segs, weights):
        """
        计算整个句子的哈希值
        :param segs: 句子中的每个单词
        :param weights: 每个单词对应的权重
        :return: 句子哈希值
        """
        scores = []

        for i, seg in enumerate(segs):
            w = weights[i]
            s = self._string_hash(seg)

            score = []
            for c in s:
                weight = 1 if c == "1" else 0
                weight *= w
                score.append(weight)
            scores.append(score)

        scores = np.array(scores).sum(0)
        result = ""
        for score in scores:
            if score > 0:
                result += "1"
            else:
                result += "0"

        return result

    @staticmethod
    def calc_distance_between_hash(hash_x, hash_y):
        """
        计算两个哈希值之间的距离，0-1表示，越接近1相似度越高
        :param hash_x: 哈希值x
        :param hash_y: 哈希值y
        :return: 相似度
        """
        min_value = min(len(hash_x), len(hash_y))
        count = 0
        for i in range(min_value):
            if hash_x[i] == hash_y[i]:
                count += 1
        return count * 1.0 / min_value
