#!/usr/bin/env python
# coding: utf-8
"""
功 能：抽取页面文本相关特征
"""

from collections import defaultdict

from hash_similarity import HashSimilarity
from paragraph_config import ParagraphFeatureConfig


class ParagraphFeatures:
    """
    文本段落相关特征类
    """
    def __init__(self, param, keywords_base_handler, logger):
        """
        初始化类变量
        :param param: 攻击模式识别参数配置
        :param keywords_base_handler: 关键词库操作对象
        :param logger: 日志对象
        """
        self.param = param
        self.keywords_base_handler = keywords_base_handler
        self.logger = logger
        self.hash_similarity_calculator = None

    def _calc_jaccard_dist(self, matched_wordbase_pos):
        """
        计算待检文本与攻击手法库的Jaccard距离
        :param matched_wordbase_pos: 已匹配词在关键词库中的下标列表
        :return: 返回字典{匹配词下标对应攻击模式库文档编号：统计次数}
        """
        # 获取已匹配关键词列表所对应的攻击模式下标
        inverse_ids = [
            self.keywords_base_handler.keywords_base[idx].get("inverse_docs")
            for idx in matched_wordbase_pos
        ]

        # 统计对应库中的各攻击模式的次数
        inverse_pattern_paragraphs = defaultdict(int)
        for paragraph_id_list in inverse_ids:
            for paragraph_id in paragraph_id_list:
                inverse_pattern_paragraphs[paragraph_id] += 1

        return inverse_pattern_paragraphs

    def filter_by_jaccard_dist(self, matched_wordbase_pos):
        """
        基于与攻击手法库的Jaccard距离进行文本过滤
        过滤思路：计算文本跟各个文档在关键词上的Jaccard系数，大于某个阈值则筛选出来
        计算方法：基于AC自动机匹配结果，只需要枚举匹配到的Keyword，计算倒排索引中paragraph_id出现次数
        :param matched_wordbase_pos: 已匹配词在关键词库中的下标列表
        :return: 基于Jaccard距离的判定标记，匹配到攻击模式的个数，匹配到攻击模式对应的关键词列表，匹配到攻击模式的下标
        """
        inverse_pattern_paragraphs = self._calc_jaccard_dist(matched_wordbase_pos)

        filtered_paragraphs_id = []
        for paragraph_id, paragraph_value in inverse_pattern_paragraphs.items():
            if paragraph_value > self.param.get("paragraph_config").get("JACCARD_THRESHOLD"):
                filtered_paragraphs_id.append((paragraph_id, paragraph_value))

        filtered_paragraphs_id = sorted(filtered_paragraphs_id, key=(lambda x: x[1]), reverse=True)

        matched_pattern_count = min(self.param.get("paragraph_config").get("TOP_INVERSE_DOCS"),
                                    len(filtered_paragraphs_id))
        self.logger.info(f"[AI Guardrail] agent pattern detection: The matched_pattern_count "
                         f"is [{matched_pattern_count}].")

        matched_paragraph_ids = [filtered_paragraphs_id[i][0] for i in range(matched_pattern_count)]
        self.logger.info(f"[AI Guardrail] agent pattern detection: The matched_paragraph_ids "
                         f"is {matched_paragraph_ids}.")

        matched_pattern_words = [self.keywords_base_handler.pattern_base[idx] for idx in matched_paragraph_ids]
        self.logger.info(f"[AI Guardrail] agent pattern detection: The matched_pattern_words "
                         f"is [{len(matched_pattern_words)}].")

        # 基于匹配攻击模式的个数判定是否攻击
        if matched_pattern_count < self.param.get("paragraph_config").get("TOP_INVERSE_DOCS"):
            self.logger.info("[AI Guardrail] agent pattern detection: matched_pattern_count "
                             "is less than TOP_INVERSE_DOCS and return Positive.")
            flag = ParagraphFeatureConfig.POSITIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return matched_pattern_count doubt.")
            flag = ParagraphFeatureConfig.DOUBT_FLAG

        return flag, matched_pattern_count, matched_pattern_words, matched_paragraph_ids

    def _calc_hash_similarity(self, content_words, content_weights, similar_paragraph_hash):
        """
        计算文本局部哈希相似度
        :param content_words: 页面所有词的列表
        :param content_weights: 页面中所有词的权重，如果非关键词或者关键词的阈值过低则置为0
        :param similar_paragraph_hash: 相似攻击模式哈希值列表
        :return: 相似度从高到低排序，文本跟攻击模式的哈希值列表
        """
        results = []
        self.hash_similarity_calculator = HashSimilarity()
        content_hash = self.hash_similarity_calculator.sentence_hash(content_words, content_weights)
        for doc_hash in similar_paragraph_hash:
            results.append(self.hash_similarity_calculator.calc_distance_between_hash(content_hash, doc_hash))

        return sorted(results, reverse=True)[0: self.param.get("paragraph_config").get("TOP_INVERSE_DOCS")]

    def filter_by_hash_similarity(self, content_words, matched_paragraph_ids):
        """
        基于局部敏感哈希相似度进行文本过滤
        :param content_words: 文本的分词列表（页面）
        :param matched_paragraph_ids: 已匹配关键词对应的攻击模式下标列表
        :return: 基于哈希相似度的判定标记，文本与攻击模式库的哈希相似度
        """
        content_keywords_weight = [
            self.keywords_base_handler.keywords_weight.get(word)
            if word in self.keywords_base_handler.keywords_weight else 0 for word in content_words
        ]
        self.logger.info(f"[AI Guardrail] agent pattern detection: The content_keywords_weight "
                         f"is [{content_keywords_weight}].")

        # 相似攻击模式的哈希值列表
        similar_paragraph_hash = [
            self.keywords_base_handler.pattern_base[idx].get("doc_hash")
            for idx in matched_paragraph_ids
        ]

        content_hash_similarity = self._calc_hash_similarity(content_words, content_keywords_weight,
                                                             similar_paragraph_hash)
        self.logger.info(f"[AI Guardrail] agent pattern detection: The content_hash_similarity "
                         f"is [{content_hash_similarity}].")

        # 基于哈希相似度判定是否攻击
        if content_hash_similarity[0] < self.param.get("paragraph_config").get("HASH_SIM_THRESHOLD_NEG"):
            self.logger.info("[AI Guardrail] agent pattern detection: content_hash_similarity "
                             "is less than HASH_SIM_THRESHOLD and return Positive.")
            flag = ParagraphFeatureConfig.POSITIVE_FLAG
        elif content_hash_similarity[0] > self.param.get("paragraph_config").get("HASH_SIM_THRESHOLD_POS"):
            self.logger.info("[AI Guardrail] agent pattern detection: content_hash_similarity "
                             "is more than HASH_SIM_THRESHOLD and return Negative.")
            flag = ParagraphFeatureConfig.NEGATIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return content_hash_similarity doubt.")
            flag = ParagraphFeatureConfig.DOUBT_FLAG

        return flag, content_hash_similarity[0: self.param.get("paragraph_config").get("TOP_DOCS")]

    @staticmethod
    def _calc_weight_similarity(content_words, similar_paragraph_weight):
        """
        计算文本权重相似度，0-1表示，越接近1表示越相似
        :param content_words: 文本的分词列表（页面）
        :param similar_paragraph_weight: 被查询文本的词列表（攻击模式库）
        :return: 权重相似度列表
        """
        result = []
        for paragraph_words, paragraph_word_weight_map in similar_paragraph_weight:
            similarity = sum([paragraph_word_weight_map[word] for word in paragraph_words if word in content_words])
            result.append(similarity / len(paragraph_words))

        return result

    def filter_by_weight_similarity(self, content_words, matched_paragraph_ids):
        """
        基于文本权重相似度进行文本过滤
        :param content_words: 文本的分词列表（页面）
        :param matched_paragraph_ids: 已匹配攻击模式下标列表
        :return: 基于权重相似度的判定标记，文本与攻击模式库的权重相似度
        """
        # 相似攻击模式的权重值列表
        similar_paragraph_weight = [
            (self.keywords_base_handler.pattern_base[idx].get("words"),
             self.keywords_base_handler.pattern_base[idx].get("word_weights_map"))
            for idx in matched_paragraph_ids
        ]

        content_weight_similarity = self._calc_weight_similarity(content_words, similar_paragraph_weight)
        self.logger.info(f"[AI Guardrail] agent pattern detection: "
                         f"The content_weight_similarity [{content_weight_similarity}].")

        # 基于文本权重相似度判定是否攻击
        if content_weight_similarity[0] < self.param.get("paragraph_config").get("WEIGHT_SIM_THRESHOLD_NEG"):
            self.logger.info("[AI Guardrail] agent pattern detection: content_weight_similarity "
                             "is less than WEIGHT_SIM_THRESHOLD and return Positive.")
            flag = ParagraphFeatureConfig.POSITIVE_FLAG
        elif content_weight_similarity[0] > self.param.get("paragraph_config").get("WEIGHT_SIM_THRESHOLD_POS"):
            self.logger.info("[AI Guardrail] agent pattern detection: content_weight_similarity "
                             "is more than WEIGHT_SIM_THRESHOLD and return Negative.")
            flag = ParagraphFeatureConfig.NEGATIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return content_weight_similarity doubt.")
            flag = ParagraphFeatureConfig.DOUBT_FLAG

        return flag, content_weight_similarity[0: self.param.get("paragraph_config").get("TOP_DOCS")]
