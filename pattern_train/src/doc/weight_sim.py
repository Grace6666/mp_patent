#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 计算文本权重相似度
修改时间: 2025/8/1 10:00
"""


def calc_weight_sim(seg_s: list[str], seg_t: list[str], weights: dict):
    """
    计算两个句子的权重相似度，0-1表示，越接近1表示越相似
    :param seg_s: 查询文本的词列表（页面）
    :param seg_t: 被查询文本的词列表（攻击模式库）
    :param weights: 查询文本的词权重
    :return: 权重相似度
    """
    dist = 0
    for seg in seg_s:
        if seg in weights:
            dist += weights[seg]
    return dist * 1.0 / len(seg_t)
