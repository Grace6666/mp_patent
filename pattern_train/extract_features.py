#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 特征提取
"""

import os
import json

from controller_feature import FastDetectorControllerFeature


def extract_pages(raw_pages):
    """
    提取页面信息
    :param raw_pages: pages
    :return: 处理后的pages信息
    """
    pages = []
    idx = 0
    for data in raw_pages:
        pages.append({
            'id': idx,
            'label': data['label'],
            'page': data['page']
        })
        idx += 1
    return pages


def extract_features(logger, config, seg_feature_extractor=None):
    """
    特征提取
    :param logger: 日志
    :param config: 配置
    :param seg_feature_extractor: 句子语法提取器
    :return:
    """
    data_config = config['data_config']
    page_filename = os.path.join(data_config['page_dir'], 'save_train_pages.json')
    with open(page_filename, 'r', encoding='utf-8') as f:
        raw_pages = json.load(f)
    pages = extract_pages(raw_pages)

    # 提取页面的特征feature
    controller = FastDetectorControllerFeature(config)
    features = []
    n = len(pages)
    for i in range(n):
        data = pages[i]
        idx = data['id']
        label = data['label']
        page = data['page']

        if seg_feature_extractor is not None:
            words, feature = controller.make_features(page, filter_switch=True)
            feature["seg_feature"] = seg_feature_extractor.extract_syntax_features(page, words)
        else:
            words, feature = controller.make_features(page)

        features.append(
            {
                'id': idx,
                'label': label,
                'page': page,
                'feature': feature
            }
        )
    logger.info("Extract feature successfully.")

    # 保存feature
    feature_config = config['feature_config']
    feature_dir = feature_config['feature_dir']
    os.makedirs(feature_dir, exist_ok=True)
    with open(os.path.join(feature_dir, 'features.json'), 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False)
    logger.info(f"Save feature to {os.path.join(feature_dir,'feature.json')} successfully")
