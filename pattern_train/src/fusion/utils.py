#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 计算召回率和误报率
修改时间: 2025/8/1 10:00
"""

from sklearn.metrics import recall_score


def calc_metric(preds, labels):
    """
    计算预测的召回率和误报率（把正常样本识别为攻击样本）
    :param preds: 预测值列表向量，1表示攻击样本，0表示正常页面
    :param labels: ground truth，格式同上
    :return: 召回率recall 误报率fp
    """
    fp, neg = 0, 0
    for label, pred in zip(labels, preds):
        if label == 0:
            neg += 1
            if pred == 1:
                fp += 1
    fp = fp * 1.0 / neg
    recall = recall_score(labels, preds)

    return recall, fp
