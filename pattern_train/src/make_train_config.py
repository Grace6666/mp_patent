#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 攻击手法训练代码配置初始化
"""
import os

import json

from model_config.model_config import read_train_config


class AttackPatternTrainConfig:
    """
    攻击手法训练参数配置
    """
    def __init__(self, logger, base_dir, flag: str, data_dir: str, output_dir: str):
        self.base_dir = base_dir
        self.flag = flag
        self.model_flag = flag
        self.develop_data_dir = data_dir[0]
        self.product_data_dir = data_dir[1]
        self.output_dir = output_dir
        self.config = {}
        self.logger = logger

    @staticmethod
    def save_config(config, filename):
        """
        存储config配置文件
        :param config: dict表示的配置文件
        :param filename: 存储的文件目录
        :return: None
        """
        with open(filename, 'w', encoding='utf-8') as json_file:
            # 配置信息序列化
            json.dump(config, json_file, ensure_ascii=False, indent=4)

    def make_init_config(self):
        """
        初始化配置文件，统一存储于config
        :return:
        """
        controller_config, db_config, data_config, fuse_config, feature_config = {}, {}, {}, {}, {}

        model_config = read_train_config(self.base_dir)

        # data_config初始化
        data_config['stopwords_filename'] = os.path.join(
            self.product_data_dir,
            model_config["data_config"]["stopwords_filename"]
        )
        data_config['attack_modes_filename'] = os.path.join(
            self.product_data_dir,
            model_config["data_config"]["attack_modes_filename"]
        )
        data_config['source_data_path'] = os.path.join(
            self.product_data_dir, model_config["data_config"]["source_data_path"])
        data_config['sample_num'] = model_config["data_config"]["sample_num"]
        data_config['page_dir'] = os.path.join(self.product_data_dir, model_config["data_config"]["page_dir"])
        data_config['syntax_patterns_dir'] = os.path.join(
            self.product_data_dir, model_config["data_config"]["syntax_patterns_dir"])
        self.config['data_config'] = data_config

        # db_config初始化
        db_config['keyword'] = model_config["db_config"]["keyword"]
        db_config['keyword_dir'] = os.path.join(
            self.product_data_dir,
            model_config["db_config"]["keyword_dir"],
            self.flag
        )
        self.config['db_config'] = db_config

        # feature_config初始化
        feature_config['keyword'] = model_config["feature_config"]["keyword"]
        feature_config['doc'] = model_config["feature_config"]["doc"]
        feature_config['fuse'] = model_config["feature_config"]["fuse"]
        feature_config['feature_dir'] = os.path.join(
            self.product_data_dir,
            model_config["feature_config"]["feature_dir"],
            self.flag
        )
        self.config['feature_config'] = feature_config

        # fuse_config初始化
        fuse_config = model_config["fuse_config"]
        fuse_config["model_dir"] = os.path.join(self.output_dir, fuse_config["model_dir"], self.model_flag)
        fuse_config["data_dir"] = os.path.join(self.product_data_dir, fuse_config["data_dir"], self.flag)
        self.config['fuse_config'] = fuse_config

        # controller_config初始化
        controller_config = {
            'base_dir': self.base_dir,
            'flag': self.flag,
            'config_dir': os.path.join(self.product_data_dir, 'config', self.flag),
            'data_config': f'data_{self.flag}.json',
            'db_config': f'db_{self.flag}.json',
            'feature_config': f'feature_{self.flag}.json',
            'fuse_config': f'fuse_{self.model_flag}.json'
        }
        self.config['controller_config'] = controller_config

    def init_config(self):
        self.make_init_config()

        # 获取配置信息
        controller_config = self.config['controller_config']
        db_config = self.config['db_config']
        fuse_config = self.config['fuse_config']
        feature_config = self.config['feature_config']
        data_config = self.config['data_config']

        # 根据配置信息拼接得到配置信息文件存储位置
        controller_config_filename = os.path.join(
            controller_config['config_dir'], f'controller_{self.flag}.json')
        data_config_filename = os.path.join(
            controller_config['config_dir'], controller_config['data_config'])
        db_config_filename = os.path.join(
            controller_config['config_dir'], controller_config['db_config'])
        feature_config_filename = os.path.join(
            controller_config['config_dir'], controller_config['feature_config'])
        fuse_config_filename = os.path.join(
            controller_config['config_dir'], controller_config['fuse_config'])

        # 写入fast_detector/config/目录下
        os.makedirs(controller_config["config_dir"], exist_ok=True)
        self.save_config(controller_config, controller_config_filename)
        self.save_config(db_config, db_config_filename)
        self.save_config(fuse_config, fuse_config_filename)
        self.save_config(feature_config, feature_config_filename)
        self.save_config(data_config, data_config_filename)

        # 日志logger
        self.logger.info('new train config generate success')
        self.logger.info(f'config :\n {self.config}')
        self.logger.info(f"saved path:{controller_config['config_dir']}")

    def update_data_config(self, flag):
        controller_config = self.config['controller_config']
        data_config_filename = os.path.join(controller_config['config_dir'], controller_config['data_config'])
        data_config = self.config['data_config']
        data_config['page_dir'] = os.path.join(data_config['page_dir'], flag)
        self.save_config(data_config, data_config_filename)

