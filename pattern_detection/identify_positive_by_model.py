#!/usr/bin/env python
# coding: utf-8
"""
功 能：基于抽取特征训练决策树分类器，初筛正常APP页面文本
"""

import os
import re
import json
import glob
import string

import jieba
import joblib
import numpy as np


from classifier_config import ClassifierConfig
from keywords_config import KeywordsFeatureConfig
from paragraph_config import ParagraphFeatureConfig
from extract_features_by_keywords import AhoCorasick
from keywords_config import KeywordsFeatureIsDetected
from paragraph_config import ParagraphFeatureIsDetected
from extract_features_by_keywords import KeywordsFeatures
from extract_features_by_paragraph import ParagraphFeatures
from extract_features_by_keywords import KeywordsBaseHandler

_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_PARAMS = {
    "keywords_config":
        {
            "KEYWORD_COUNT_IS_DETECTED": KeywordsFeatureIsDetected.KEYWORD_COUNT_IS_DETECTED,
            "KEYWORD_BAYES_PROB_IS_DETECTED": KeywordsFeatureIsDetected.KEYWORD_BAYES_PROB_IS_DETECTED,
            "KEYWORD_DIST_IS_DETECTED": KeywordsFeatureIsDetected.KEYWORD_DIST_IS_DETECTED,
            "KEYWORD_MATCH_THRESHOLD_NEG": KeywordsFeatureConfig.KEYWORD_MATCH_THRESHOLD.get("neg"),
            "KEYWORD_MATCH_THRESHOLD_POS": KeywordsFeatureConfig.KEYWORD_MATCH_THRESHOLD.get("pos"),
            "BAYES_PROB_THRESHOLD_NEG": KeywordsFeatureConfig.BAYES_PROB_THRESHOLD.get("neg"),
            "BAYES_PROB_THRESHOLD_POS": KeywordsFeatureConfig.BAYES_PROB_THRESHOLD.get("pos"),
            "BAYES_NGRAM": KeywordsFeatureConfig.BAYES_NGRAM,
            "BAYES_NGRAM_MARGIN": KeywordsFeatureConfig.BAYES_NGRAM_MARGIN,
            "WEIGHT_THRESHOLD": KeywordsFeatureConfig.WEIGHT_THRESHOLD,
            "QT_POS_RATE": KeywordsFeatureConfig.QT_POS_RATE,
            "QT_CHECK_POS": KeywordsFeatureConfig.QT_CHECK_POS,
            "QT_CHECK_THRESHOLD": KeywordsFeatureConfig.QT_CHECK_THRESHOLD,
        },
    "paragraph_config":
        {
            "MATCHED_PATTERN_IS_DETECTED": ParagraphFeatureIsDetected.MATCHED_PATTERN_IS_DETECTED,
            "WEIGHT_SIMILARITY_IS_DETECTED": ParagraphFeatureIsDetected.HASH_SIMILARITY_IS_DETECTED,
            "HASH_SIMILARITY_IS_DETECTED": ParagraphFeatureIsDetected.HASH_SIMILARITY_IS_DETECTED,
            "JACCARD_THRESHOLD": ParagraphFeatureConfig.JACCARD_THRESHOLD,
            "TOP_INVERSE_DOCS": ParagraphFeatureConfig.TOP_INVERSE_DOCS,
            "WINDOW_SIZE": ParagraphFeatureConfig.WINDOW_SIZE,
            "HASH_SIM_THRESHOLD_NEG": ParagraphFeatureConfig.HASH_SIM_THRESHOLD.get("neg"),
            "HASH_SIM_THRESHOLD_POS": ParagraphFeatureConfig.HASH_SIM_THRESHOLD.get("pos"),
            "WEIGHT_SIM_THRESHOLD_NEG": ParagraphFeatureConfig.WEIGHT_SIM_THRESHOLD.get("neg"),
            "WEIGHT_SIM_THRESHOLD_POS": ParagraphFeatureConfig.WEIGHT_SIM_THRESHOLD.get("pos"),
            "TOP_DOCS": ParagraphFeatureConfig.TOP_DOCS,
        },
    "classifier_config":
        {
            "CLASSIFIER_THRESHOLD_NEG": ClassifierConfig.CLASSIFIER_THRESHOLD_NEG,
            "CLASSIFIER_THRESHOLD_POS": ClassifierConfig.CLASSIFIER_THRESHOLD_POS,
            "CONTENT_LENGTH_THRESHOLD": ClassifierConfig.CONTENT_LENGTH_THRESHOLD
        },
    }


class PatternIdentification:
    """
    攻击手法识别类
    """
    def __init__(self, logger):
        self.logger = logger
        self.keywords_base_handler = None
        self._tokenizer = None
        # 参数配置
        self.param = None
        self.ac_matcher = None
        self.classifier = None
        self.keyword_features_extractor = None
        self.paragraph_features_extractor = None

    def init_main(self):
        """
        初始化主函数
        """
        self.keywords_base_handler = KeywordsBaseHandler(self.logger)
        self.keywords_base_handler.init()

        # Loading jieba cost 0.5s
        self._init_segmenter()

        # 加载参数配置
        self.param = self._init_param()

        # 加载AC自动机模型
        keywords_base = self.keywords_base_handler.keywords_base
        self.ac_matcher = AhoCorasick(keywords_base, self.logger)

        # 加载检测模型
        self._init_classifier()

        self.keyword_features_extractor = KeywordsFeatures(self.param, self.logger)
        self.paragraph_features_extractor = ParagraphFeatures(self.param, self.keywords_base_handler, self.logger)


    def _init_classifier(self):
        """
        初始化检测模型
        """
        try:
            model_dir_wildcard = os.path.join(_BASE_DIR, "model/AttackPatternDetectionModel_*")
            matched_paths = glob.glob(model_dir_wildcard)

            model_path = os.path.realpath(os.path.join(matched_paths[0], ClassifierConfig.MODEL_PATH))
            with (os.fdopen(os.open(model_path, ClassifierConfig.FLAGS, ClassifierConfig.MODE), "rb")
                  as model_file):
                self.classifier = joblib.load(model_file)

            self.logger.info("[AI Guardrail] agent pattern detection: The model "
                             "for pattern identification loaded successfully.")

        except FileNotFoundError:
            self.logger.error("[AI Guardrail] agent pattern detection: The model for pattern identification not found.")
            raise
        except Exception as e:
            self.logger.error("[AI Guardrail] agent pattern detection: The model "
                              "for pattern identification loaded failed.")
            raise


    def _init_segmenter(self):
        """
        初始化分词器
        """
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.initialize()
        self._tokenizer.cut("攻击模式识别")
        self.logger.info("[AI Guardrail] agent pattern detection: The jieba tokenizer loaded successfully.")

    def _init_param(self):
        """
        初始化参数配置
        """
        # 如果加载配置文件中发生异常，确保加载默认参数
        json_data = _DEFAULT_PARAMS

        model_dir_wildcard = os.path.join(_BASE_DIR, "model/AttackPatternDetectionModel_*")
        matched_paths = glob.glob(model_dir_wildcard)
        param_path = os.path.realpath(os.path.join(matched_paths[0], "config/attack_pattern_config.json"))
        try:
            with os.fdopen(
                    os.open(param_path, ClassifierConfig.FLAGS, ClassifierConfig.MODE), "r", encoding="utf-8"
            ) as json_file:
                json_data = json.load(json_file)
                # 判断配置是否完整
                json_integrity_flag, json_field = self._check_config_integrity(json_data)
                if not json_integrity_flag:
                    self.logger.error(f"[AI Guardrail] agent pattern detection: attack_pattern_config.json "
                                      f"miss correct {json_field}, load default parameters.")
                    raise RuntimeError("Miss correct parameters.")

                # 检查配置参数字典必要字段与值类型
                param_format_flag, param_field = self._check_attack_pattern_params(json_data)
                if not param_format_flag:
                    self.logger.error(f"[AI Guardrail] agent pattern detection: config/attack_pattern_config.json "
                                      f"miss correct {param_field}, load default parameters.")
                    raise RuntimeError("Miss correct parameters.")

            self.logger.info("[AI Guardrail] agent pattern detection: config/attack_pattern_config.json "
                             "loaded successfully.")

        except FileNotFoundError:
            self.logger.error("[AI Guardrail] agent pattern detection: config/attack_pattern_config.json not found,"
                              " load default parameters.")
            raise
        except json.JSONDecodeError:
            self.logger.error("[AI Guardrail] agent pattern detection: config/attack_pattern_config.json "
                              "parsed failed, load default parameters.")
            raise
        except Exception as e:
            self.logger.error(f"[AI Guardrail] agent pattern detection: config/attack_pattern_config.json "
                              f"loaded failed, load default parameters.")
            raise

        return json_data
    
    @staticmethod
    def _check_config_integrity(json_data):
        """
        判断配置是否完整
        :param json_data: 待检配置参数字典
        :return: 格式是否符合预期，异常项
        """
        json_format_flag, json_field = True, None

        for config_item in ["keywords_config", "paragraph_config", "classifier_config"]:
            if config_item not in json_data:
                return False, config_item

        return json_format_flag, json_field

    def _check_attack_pattern_params(self, json_data):
        """
        检查配置参数字典必要字段与值类型
        :param json_data: 待检配置参数字典
        :return: 格式是否符合预期，异常字段
        """
        config_format_flag, json_field = True, None
        # 判断各配置项键是否完整，值类型是否符合预期
        keywords_config_map = {
            "KEYWORD_COUNT_IS_DETECTED": bool,
            "KEYWORD_BAYES_PROB_IS_DETECTED": bool,
            "KEYWORD_DIST_IS_DETECTED": bool,
            "KEYWORD_MATCH_THRESHOLD_NEG": int,
            "KEYWORD_MATCH_THRESHOLD_POS": int,
            "BAYES_PROB_THRESHOLD_NEG": float,
            "BAYES_PROB_THRESHOLD_POS": float,
            "BAYES_NGRAM": int,
            "BAYES_NGRAM_MARGIN": int,
            "WEIGHT_THRESHOLD": int,
            "QT_POS_RATE": list,
            "QT_CHECK_POS": list,
            "QT_CHECK_THRESHOLD": list,
        }
        paragraph_config_map = {
            "MATCHED_PATTERN_IS_DETECTED": bool,
            "WEIGHT_SIMILARITY_IS_DETECTED": bool,
            "HASH_SIMILARITY_IS_DETECTED": bool,
            "JACCARD_THRESHOLD": int,
            "TOP_INVERSE_DOCS": int,
            "WINDOW_SIZE": int,
            "HASH_SIM_THRESHOLD_NEG": float,
            "HASH_SIM_THRESHOLD_POS": float,
            "WEIGHT_SIM_THRESHOLD_NEG": float,
            "WEIGHT_SIM_THRESHOLD_POS": float,
            "TOP_DOCS": int,
        }
        classifier_config_map = {
            "CLASSIFIER_THRESHOLD_NEG": float,
            "CLASSIFIER_THRESHOLD_POS": float,
            "CONTENT_LENGTH_THRESHOLD": int,
        }
        keyword_flag, param_item = self._check_key_and_type(json_data, keywords_config_map, "keywords_config")
        if not keyword_flag:
            return keyword_flag, param_item

        paragraph_flag, param_item = self._check_key_and_type(json_data, paragraph_config_map, "paragraph_config")
        if not paragraph_flag:
            return paragraph_flag, param_item

        classifier_flag, param_item = self._check_key_and_type(json_data, classifier_config_map, "classifier_config")
        if not classifier_flag:
            return classifier_flag, param_item

        return config_format_flag, json_field

    @staticmethod
    def _check_key_and_type(json_data, config_map, key):
        """
        检查param项是否符合预期
        :param json_data: 待检配置参数字典
        :param config_map: 待检查param项对象
        :param key: 待检查项名称
        :return: 格式是否符合预期，异常字段
        """
        for param_item, param_type in config_map.items():
            if param_item not in json_data.get(key):
                return False, param_item
            if not isinstance(json_data.get(key).get(param_item), param_type):
                return False, param_item

        return True, None

    def _cut_words(self, text):
        """
        jieba分词，并删除分词结果中的符号
        :param text: 待切分文本
        :return: 经分词后的列表
        """
        template = ParagraphFeatureConfig.SPLIT_TEMPLATE

        results = []
        for seg in self._tokenizer.cut(text):
            if seg not in string.punctuation and seg not in self.keywords_base_handler.stop_words_base \
                    and len(re.findall(template, seg)) == 0:
                results.append(seg)

        return results

    @staticmethod
    def _map_detection_signal(is_attack_detected, cur_feature_flag):
        """
        基于特征是否检出攻击的配置、当前特征判断结果，映射检测信号
        :param is_attack_detected: 是否检出攻击的配置
        :param cur_feature_flag: 当前特征过滤标记
        :return: 映射检测信号
        """
        if cur_feature_flag != 1:
            return cur_feature_flag
        # 该特征过滤标记为1，且特征配置检出攻击，则返回检测信号为1，即攻击
        elif is_attack_detected is True:
            return 1
        # 该特征过滤标记为1，但特征配置不检出攻击，则返回检测信号为-1，即存疑
        elif is_attack_detected is False:
            return -1

    def _classify_by_decision_tree(self, selection_features):
        """
        定义决策树分类器
        :param selection_features: 快筛特征
        :return: 决策树判定标记
        """
        # 获取模型预测结果
        model_pred = self.classifier.predict_proba(selection_features)[0, 1]

        if model_pred < self.param.get("classifier_config").get("CLASSIFIER_THRESHOLD_NEG"):
            self.logger.info("[AI Guardrail] agent pattern detection: model_pred "
                             "is less than CLASSIFIER_THRESHOLD_NEG and return Positive.")
            return ClassifierConfig.POSITIVE_FLAG
        elif model_pred > self.param.get("classifier_config").get("CLASSIFIER_THRESHOLD_POS"):
            self.logger.info("[AI Guardrail] agent pattern detection: model_pred "
                             "is more than CLASSIFIER_THRESHOLD_POS and return Negative.")
            return ClassifierConfig.NEGATIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return model_pred doubt.")
            return ClassifierConfig.DOUBT_FLAG

    def _preprocess_content(self, raw_content):
        """
        待检文本预处理：1.超长文本截断；2.去除停用词
        :param raw_content: 原始待检文本
        :return: 预处理文本
        """
        # 超长文本截断
        content_length = len(raw_content)
        _content_length_threshold = self.param.get("classifier_config").get("CONTENT_LENGTH_THRESHOLD")
        if content_length > _content_length_threshold:
            raw_content = raw_content[: _content_length_threshold]
            self.logger.warning(f"[AI Guardrail] agent pattern detection: content length {content_length}. "
                                f"Beyond length threshold {_content_length_threshold}. Truncate content.")

        # 去除停用词
        content_words = self._cut_words(raw_content)
        content = "".join(content_words)

        return content, content_words

    def filter_by_extract_features(self, content):
        """
        抽取当前文本特征，并基于特征实现文本多步骤过滤
        :param content: 待检文本
        :return: 判定标记
        """
        content, content_words = self._preprocess_content(content)

        # 调用KeywordsFeatures类方法，抽取页面文本关键词
        matched_keywords_pos, matched_wordbase_pos, matched_keywords = self.ac_matcher.match_keyword_by_ac(content)

        # 关键词数量
        keyword_count_flag = self.keyword_features_extractor.filter_by_keywords_count(matched_keywords)
        keyword_count_is_detected = self.param.get("keywords_config").get("KEYWORD_COUNT_IS_DETECTED")
        detection_signal = self._map_detection_signal(keyword_count_is_detected, keyword_count_flag)
        if detection_signal != -1:
            return detection_signal

        # 关键词贝叶斯概率
        keywords_bayes_prob_flag, keywords_bayes_prob = self.keyword_features_extractor.filter_by_bayes_prob(
            matched_keywords, matched_keywords_pos, self.keywords_base_handler)
        keyword_bayes_prob_is_detected = self.param.get("keywords_config").get("KEYWORD_BAYES_PROB_IS_DETECTED")
        detection_signal = self._map_detection_signal(keyword_bayes_prob_is_detected, keywords_bayes_prob_flag)
        if detection_signal != -1:
            return detection_signal

        # 关键词距离
        keywords_dist_flag, keywords_dist = self.keyword_features_extractor.filter_by_keywords_dist(
            matched_keywords_pos)
        keyword_dist_is_detected = self.param.get("keywords_config").get("KEYWORD_DIST_IS_DETECTED")
        detection_signal = self._map_detection_signal(keyword_dist_is_detected, keywords_dist_flag)
        if detection_signal != -1:
            return detection_signal

        # 抽取页面文本jaccard距离特征
        matched_pattern_flag, matched_pattern_count, matched_pattern_words, matched_paragraph_ids = \
            self.paragraph_features_extractor.filter_by_jaccard_dist(matched_wordbase_pos)
        matched_pattern_is_detected = self.param.get("paragraph_config").get("MATCHED_PATTERN_IS_DETECTED")
        detection_signal = self._map_detection_signal(matched_pattern_is_detected, matched_pattern_flag)
        if detection_signal != -1:
            return detection_signal

        # 抽取页面文本权重相似度特征
        weight_similarity_flag, content_weight_similarity = \
            self.paragraph_features_extractor.filter_by_weight_similarity(content_words, matched_paragraph_ids)
        weight_similarity_is_detected = self.param.get("paragraph_config").get("WEIGHT_SIMILARITY_IS_DETECTED")
        detection_signal = self._map_detection_signal(weight_similarity_is_detected, weight_similarity_flag)
        if detection_signal != -1:
            return detection_signal

        # 抽取页面局部哈希相似度特征
        hash_similarity_flag, content_hash_similarity = \
            self.paragraph_features_extractor.filter_by_hash_similarity(content_words, matched_paragraph_ids)
        hash_similarity_is_detected = self.param.get("paragraph_config").get("HASH_SIMILARITY_IS_DETECTED")
        detection_signal = self._map_detection_signal(hash_similarity_is_detected, hash_similarity_flag)
        if detection_signal != -1:
            return detection_signal

        # 已删除keywords_bayes_prob特征，并替换模型文件
        selection_features = [len(matched_keywords), keywords_dist.get("mean")]
        selection_features.extend(keywords_dist.get("qt"))
        selection_features.extend(content_hash_similarity)
        selection_features.extend(content_weight_similarity)

        # 决策树做最终判定
        selection_features = np.expand_dims(np.array(selection_features), axis=0)
        res = self._classify_by_decision_tree(selection_features)

        return res
