#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 融合评估模型训练
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fusion.utils import calc_metric


def load_data(data_dir):
    """
    加载数据集和标签
    :param data_dir: 数据所在目录
    :return: 输入数据inputs、标签labels
    """
    inputs = np.loadtxt(os.path.join(data_dir, 'input.txt'))
    labels = np.loadtxt(os.path.join(data_dir, 'label.txt')).astype(np.int32)
    return inputs, labels


def split_data(inputs, labels, test_size, random_state):
    """
    将数据集划分为训练集、验证集、测试机
    :param inputs: 输入数据
    :param labels: 标签
    :param test_size: 测试集和验证集占总数居的比例
    :param random_state: 随机种子，确保分割的可重复性
    :return: train训练集、val验证集、test测试集
    """
    train_x, test_x, train_y, test_y = train_test_split(
        inputs,
        labels,
        test_size=test_size,
        random_state=random_state
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x,
        train_y,
        test_size=test_size,
        random_state=random_state
    )
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def train_and_evaluate_model(train, val, model):
    """
    训练随机森林模型，并在训练集和验证集上进行评估
    :param train: 训练集
    :param val: 验证集
    :param model: 随机森林模型实例
    :return: model训练好的随机森林模型、训练集指标、验证集指标
    """
    model.fit(train[0], train[1])
    pred_train_y = model.predict(train[0])
    pred_val_y = model.predict(val[0])

    recall_train, fp_train = calc_metric(pred_train_y, train[1])
    recall_val, fp_val = calc_metric(pred_val_y, val[1])

    return model, (recall_train, fp_train), (recall_val, fp_val)


def save_model(model, model_path):
    """
    保存训练好的模型
    :param model: 训练好的模型
    :param model_path: 保存模型的文件路径
    :return: None
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def evaluate_thresholds(model, train, test, thresholds, model_dir):
    """
    在不同的概率阈值下评估模型的性能，并将结果保存到文件中
    :param model: 训练好的模型
    :param train: 训练集
    :param test: 测试集
    :param thresholds: 概率阈值列表
    :param model_dir: 保存结果文件的目录
    :return: None
    """
    prob_train = model.predict_proba(train[0])[:, 1]
    prob_test = model.predict_proba(test[0])[:, 1]

    with open(os.path.join(model_dir, 'result.txt'), 'w') as f:
        for t in thresholds:
            pred_train_y = (prob_train > t).astype(int)
            pred_test_y = (prob_test > t).astype(int)

            recall_train, fp_train = calc_metric(pred_train_y, train[1])
            recall_test, fp_test = calc_metric(pred_test_y, test[1])

            f.write('threshold=%.2f, train recall=%.2f, fp=%.2f\n' % (t, 100 * recall_train, 100 * fp_train))
            f.write('threshold=%.2f, test recall=%.2f, fp=%.2f\n' % (t, 100 * recall_test, 100 * fp_test))


def train_rf(logger, fuse_config, base_dir):
    """
    融合评估模型训练
    :param logger: 日志
    :param fuse_config: 融合评估模型配置文件
    :param base_dir: 工作目录
    :return: None
    """

    data_dir = os.path.join(base_dir, fuse_config['data_dir'])
    model_dir = os.path.join(base_dir, fuse_config['model_dir'])
    os.makedirs(model_dir, exist_ok=True)

    inputs, labels = load_data(data_dir)
    logger.info(f"Load data successfully. Total samples: {len(inputs)}")
    train, val, test = split_data(inputs, labels, fuse_config['test_size'], fuse_config['random_state'])
    logger.info(
        f"Split data successfully. Train samples: {len(train)}, Val samples: {len(val)}, Test samples: {len(test)}."
    )

    if not all([train, val, test]):
        logger.error("The training, validation, or test data is empty.")
        raise ValueError("The training, validation, or test data is empty.")

    model = RandomForestClassifier(max_depth=fuse_config['max_depth'], n_estimators=fuse_config['n_estimators'])
    model, train_metrics, val_metrics = train_and_evaluate_model(train, val, model)
    logger.info(f"train recall and fp: {100 * train_metrics[0]}, {100 * train_metrics[1]}")
    logger.info(f"val recall and fp: {100 * val_metrics[0]}, {100 * val_metrics[1]}")

    model_filename = os.path.join(model_dir, "model.pkl")
    save_model(model, model_filename)
    logger.info(f"save model at {model_filename}")

    thresholds = np.arange(0.05, 1.0, 0.05).tolist()
    evaluate_thresholds(model, train, test, thresholds, model_dir)
    logger.info(f"the result of evaluate thresholds save at {os.path.join(model_dir, 'result.txt')}")
