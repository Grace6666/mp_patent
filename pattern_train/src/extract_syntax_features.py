#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 句子语法特征提取
"""

import re
import os
import json


class FeatureExtractorConfig:
    """
    特征提取配置类
    """
    def __init__(self, language, logger, data_config):
        self.language = language
        self.logger = logger
        self.syntax_patterns_dir = os.path.join(str(data_config["syntax_patterns_dir"]), self.language)

        patterns = (
            "punctuations", "deception_patterns", "profanity_patterns",
            "politeness_patterns", "ambiguity_patterns", "negative_patterns", "emotional_patterns",
            "evasive_patterns", "jailbreak_patterns", "persona_patterns",
            "false_safety_patterns", "misinformation_patterns", "fake_expertise_patterns",
            "leading_question_patterns", "cognitive_load_patterns", "negative_tone_patterns", "social_patterns"
        )
        code_pattern = {
            r"`{3}[\s\S]*?`{3}",
            r"`[^`]*`",
            r"<[^>]+>",
            r"\bdef\b",
            r"\bimport\b",
            r"\bclass\b",
            r"\bconsole\.log\b",
            r"\bpublic\s+class\b",
            r"\bSELECT\b\s+.*\s+FROM\b",
            r"\bfunction\b\s+\w+\("
        }

        self.punctuations = set()
        self.deception_patterns = set()
        self.profanity_patterns = set()
        self.politeness_patterns, self.politeness_patterns_regex = set(), list()
        self.ambiguity_patterns, self.ambiguity_patterns_regex = set(), list()
        self.negative_patterns, self.negative_patterns_regex = set(), list()
        self.emotional_patterns, self.emotional_patterns_regex = set(), list()
        self.evasive_patterns, self.evasive_patterns_regex = set(), list()
        self.jailbreak_patterns, self.jailbreak_patterns_regex = set(), list()
        self.persona_patterns, self.persona_patterns_regex = set(), list()
        self.false_safety_patterns, self.false_safety_patterns_regex = set(), list()
        self.misinformation_patterns, self.misinformation_patterns_regex = set(), list()
        self.fake_expertise_patterns, self.fake_expertise_patterns_regex = set(), list()
        self.leading_question_patterns, self.leading_question_patterns_regex = set(), list()
        self.cognitive_load_patterns, self.cognitive_load_patterns_regex = set(), list()
        self.negative_tone_patterns, self.negative_tone_patterns_regex = set(), list()
        self.social_patterns, self.social_patterns_regex = set(), list()

        try:
            # 知识库加载和正则表达式编译都可能出现错误
            self._load_patterns(patterns)
            self.word_freq = self._load_json("word_freq.json")
            self.code_patterns = self._merge_regex_patterns(code_pattern)
        except Exception as e:
            self.logger.error("Load feature extraction syntax lib error.")
            raise RuntimeError("Load feature extraction syntax lib error") from e

    def _load_json(self, file_path):
        """
        加载JSON文件
        """
        real_path = os.path.realpath(os.path.join(self.syntax_patterns_dir, file_path))
        try:
            with open(real_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data
        except Exception as e:
            self.logger.error("Load {file_path} failed.")
            raise

    @staticmethod
    def _format_pattern(pattern):
        """
        格式化语法模式库中的字符串和正则表达式
        :param pattern: 字符串或正则表达式对象
        :returns: 格式化后的正则表达式字符串
        """
        if isinstance(pattern, str):
            base = pattern
        elif isinstance(pattern, re.Pattern):
            base = pattern.pattern
        else:
            raise RuntimeError(f"Unsupported pattern type: {type(pattern).__name__}")
        return base

    @staticmethod
    def _phrase_regex(phrases, env_lang):
        """
        将语法模式库中的短语转换为正则表达式对象
        :param phrases: 包含短语的列表或集合
        :returns: 包含正则表达式对象的列表
        """
        patterns = []
        for phrase in phrases:
            escaped = re.escape(phrase)
            if env_lang == "cn":
                pattern = re.compile(escaped, re.UNICODE)
            else:
                pattern = re.compile(escaped, re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _load_patterns(self, patterns):
        """
        加载语法模式库中的单词和正则表达式
        :param patterns: 包含单词和正则表达式对象的列表
        """
        for pattern_name in patterns:
            file_path = self._get_pattern_file_path(pattern_name)
            data = self._load_json(file_path)
            words = data.get("words", [])
            phrases = data.get("phrases", [])
            terms = next(iter(data.values()), list())
            words_set = set(words)

            if not words and not phrases and terms:
                terms_set = set(terms)
                self._set_words(pattern_name, terms_set)
                self._set_phrases_regex(f"{pattern_name}_regex", terms_set)
                continue

            if words:
                self._set_words(pattern_name, words_set)
            if phrases:
                self._set_phrases_regex(f"{pattern_name}_regex", set(phrases))

    def _merge_regex_patterns(self, patterns):
        """
        合并语法模式库中的单词和正则表达式为一个正则表达式对象，匹配更快速
        :param patterns: 包含单词和正则表达式对象的列表
        :returns: 合并后的正则表达式对象
        """
        combined_pattern = "|".join(self._format_pattern(p) for p in patterns)
        return re.compile(combined_pattern, re.IGNORECASE)

    def _get_pattern_file_path(self, pattern_name):
        """
        构建模式文件路径
        :param pattern_name: 模式名称
        :returns: 模式文件路径
        """
        filename = f"{pattern_name}.json"
        return os.path.realpath(os.path.join(self.syntax_patterns_dir, filename))

    def _set_words(self, attr_name, words_set):
        """
        设置 words 属性（小写）
        :param attr_name: 属性名称
        :param words_set: 单词集合
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, words_set)

    def _set_phrases_regex(self, attr_name, phrases):
        """
        设置 phrases 的正则表达式属性
        :param attr_name: 属性名称
        :param phrases: 短语集合
        """
        if hasattr(self, attr_name):
            phrase_regex = self._phrase_regex(set(phrases), self.language)
            combined_patterns = self._merge_regex_patterns(phrase_regex)
            setattr(self, attr_name, combined_patterns)


class GeneralFeatureExtractor:
    """
    通用特征提取基类
    """
    @staticmethod
    def compute_sentence_length(words):
        """
        计算句子长度
        :param words: 句子中的词列表
        :returns: 句子长度（词数）
        """
        return len(words)

    @staticmethod
    def compute_avg_token_length(words):
        """
        计算平均词长度
        :param words: 句子中的词列表
        :returns: 平均词长度（字符数）
        """
        return sum(len(word) for word in words) / len(words) if len(words) != 0 else 0

    @staticmethod
    def compute_lexical_diversity_score(words):
        """
        计算词汇多样性（不同词的数量 / 总词数）
        :param words: 词列表
        :returns: 词汇多样性得分
        """
        unique_words = set(words)

        return len(unique_words) / len(words) if len(words) != 0 else 0


class SpecificFeatureExtractor:
    """
    特定特征提取类
    """
    def __init__(self, feature_extractor_config):
        self.feature_extractor_config = feature_extractor_config

    @staticmethod
    def compute_matching_score(text, words, word_patterns, combined_regex):
        """
        计算 token 列表和语法模式库内容的匹配分数
        :param text: 输入文本
        :param words: token 列表
        :param word_patterns: 语法模式库中的单词列表
        :param combined_regex: 语法模式库中的正则表达式列表
        :return: 匹配分数, [0, 1]
        """
        total_words = len(words)
        if total_words == 0:
            return 0
        word_matches = sum(1 for word in words if word in word_patterns) if word_patterns else 0
        re_matches = len(combined_regex.findall(text)) if combined_regex else 0
        matched_terms = word_matches + re_matches

        return min(matched_terms / total_words, 1.0)

    def compute_avg_word_frequency(self, words):
        """
        计算平均词频（所有词的词频总和 / 总词数）
        :param words: 词列表
        :returns: 平均词频
        """
        if len(words) == 0:
            return 0
        total_freq = sum(self.feature_extractor_config.word_freq.get(word, 0) for word in words)
        avg_freq = total_freq / len(words)
        return avg_freq

    def compute_punctuation_ratio(self, text):
        """
        计算标点符号的比例（标点符号数量 / 文本长度）
        :param text: 输入文本（字符串）
        :returns: 标点符号的比例
        """
        text_length = len(text)
        punctuation_count = sum(1 for char in text if char in self.feature_extractor_config.punctuations)
        punctuation_ratio = punctuation_count / text_length if text_length > 0 else 0
        return punctuation_ratio

    def compute_social_term_frequency(self, text, words):
        """
        计算社交术语频率（匹配到的社交术语数量 / 总词数）
        :param text: 输入文本（字符串）
        :param words: 词列表
        :returns: 社交术语频率
        """
        return self.compute_matching_score(text, words, self.feature_extractor_config.social_patterns,
                                           self.feature_extractor_config.social_patterns_regex)

    def compute_deception_score(self, words):
        """
        计算欺骗性得分（1 - 匹配到的欺骗性词汇数量 / 总词数）
        :param words: 词列表
        :returns: 欺骗性得分
        """
        if len(words) == 0:
            return 0
        matched_terms = sum(1 for word in words if word in self.feature_extractor_config.deception_patterns)
        deception_score = 1 - (matched_terms / len(words))
        return deception_score

    def compute_politeness_score(self, text, words):
        """
        计算礼貌性得分（匹配到的礼貌性词汇数量 / 总词数）
        :param text: 输入文本（字符串）
        :param words: 词列表
        :returns: 礼貌性得分
        """
        return self.compute_matching_score(text, words, self.feature_extractor_config.politeness_patterns,
                                           self.feature_extractor_config.politeness_patterns_regex)

    def compute_cognitive_load_score(self, text, words):
        """
        计算认知负载得分（匹配到的认知负载词汇数量 / 总词数）
        :param text: 输入文本（字符串）
        :param words: 词列表
        :returns: 认知负载得分
        """
        return self.compute_matching_score(text, words, self.feature_extractor_config.cognitive_load_patterns,
                                           self.feature_extractor_config.cognitive_load_patterns_regex)


class FeatureExtract:
    """
    句法特征提取类
    """
    def __init__(self, logger, data_config):
        feature_extractor_config = FeatureExtractorConfig("cn", logger, data_config)
        self.extractor_general = GeneralFeatureExtractor()
        self.extractor_specific = SpecificFeatureExtractor(feature_extractor_config)

    def extract_syntax_features(self, text, words):
        """
        提取句法特征
        """
        features = {
            "avg_token_length": self.extractor_general.compute_avg_token_length(words),
            "sentence_length": self.extractor_general.compute_sentence_length(words),
            "lexical_diversity_score": self.extractor_general.compute_lexical_diversity_score(words),

            "avg_word_frequency": self.extractor_specific.compute_avg_word_frequency(words),
            "social_word_frequency": self.extractor_specific.compute_social_term_frequency(text, words),
            "punctuation_ratio": self.extractor_specific.compute_punctuation_ratio(text),
            "deception_score": self.extractor_specific.compute_deception_score(words),
            "politeness_score": self.extractor_specific.compute_politeness_score(text, words),
            "cognitive_load_score": self.extractor_specific.compute_cognitive_load_score(text, words),
        }

        return features
