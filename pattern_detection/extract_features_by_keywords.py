
"""
功 能：抽取关键词库相关特征
"""

import os
import json
import glob
from collections import defaultdict

import ahocorasick

from hash_similarity import HashSimilarity
from keywords_config import FilePermissions
from paragraph_config import PatternBaseConfig
from keywords_config import KeywordsBaseConfig
from keywords_config import KeywordsFeatureConfig

_BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class AhoCorasick:
    """
    AC自动机
    """

    def __init__(self, keywords_base, logger):
        """
        AC自动机初始化
        :param keywords_base: 关键词文本列表
        :param logger: 日志对象
        """
        self.ac_tree = ahocorasick.Automaton()
        self.keyword_base = keywords_base
        for index, keyword_body in enumerate(keywords_base):
            word = keyword_body.get("word")
            self.ac_tree.add_word(word, (index, word))
        logger.info(f"[AI Guardrail] agent pattern detection: The ac-tree loaded {len(keywords_base)} words.")
        self.ac_tree.make_automaton()

    def match_keyword_by_ac(self, sentence):
        """
        基于AC自动机匹配关键词
        :param sentence: 待匹配文本
        :return: 匹配元组（在sentence中的位置，在关键词库中的下标，关键词）
        """
        matched_keywords_pos, matched_wordbase_pos, matched_keywords = [], [], []
        for res in self.ac_tree.iter(sentence):
            matched_keywords_pos.append(res[0])
            matched_wordbase_pos.append(res[1][0])
            matched_keywords.append(res[1][1])

        return matched_keywords_pos, matched_wordbase_pos, matched_keywords


class KeywordsBaseHandler:
    """
    关键词库
    """

    def __init__(self, logger):
        """
        初始化关键词列表
        """
        self.keywords_weight = None
        self.keywords_id = None
        self.keywords_base = None
        self.hash_similarity_calculator = None
        self.stop_words_base = None
        self.keywords_bayes_prob_base = None
        self.pattern_base = None
        self.logger = logger

    def init(self):
        # 关键词表
        self.keywords_base, self.keywords_id, self.keywords_weight = self._load_keywords_base()
        # 关键词贝叶斯概率表
        self.keywords_bayes_prob_base = self._load_bayes_prob_base()
        # 停用词表
        self.stop_words_base = self._load_stop_words_base()

        # 文本哈希相似度计算
        self.hash_similarity_calculator = HashSimilarity()
        # 攻击手法库
        self.pattern_base = self._load_pattern_base()

    def _read_json_file(self, file_path):
        """
        读取json文件
        :param file_path: 文件路径
        :return: 读取到的json字典
        """
        json_data = {}
        try:
            model_dir_wildcard = os.path.join(_BASE_DIR, "model/AttackPatternDetectionModel_*")
            matched_paths = glob.glob(model_dir_wildcard)

            file_real_path = os.path.realpath(os.path.join(matched_paths[0], file_path))
            with os.fdopen(
                    os.open(file_real_path, FilePermissions.FLAGS, FilePermissions.MODE), "r", encoding="utf-8"
            ) as json_file:
                json_data = json.load(json_file)
            self.logger.info(f"[AI Guardrail] agent pattern detection: {file_path} loaded successfully.")

        except FileNotFoundError:
            self.logger.error(f"[AI Guardrail] agent pattern detection: {file_path} not found.")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"[AI Guardrail] agent pattern detection: {file_path} parsed failed.")
            raise
        except Exception as e:
            self.logger.error(f"[AI Guardrail] agent pattern detection: {file_path} loaded failed.")
            raise

        return json_data

    def _load_keywords_base(self):
        """
        加载关键词库
        :return: 关键词表，各关键词逆文档编号，关键词权重字典
        """
        keywords_base = []
        keywords_id = {}
        keywords_weight = {}

        keyword_base_file = self._read_json_file(KeywordsBaseConfig.KEYWORDS_BASE_PATH)
        for keyword_body in keyword_base_file:
            # 键idx，word，weight，inverse_docs
            keyword_body["bayes_prob"] = 0.0
            keywords_base.append(keyword_body)
            keywords_id[keyword_body.get("word")] = keyword_body.get("idx")
            keywords_weight[keyword_body.get("word")] = keyword_body.get("weight")
        self.logger.info(f"[AI Guardrail] agent pattern detection: The keyword base loaded {len(keywords_base)} lines.")

        return keywords_base, keywords_id, keywords_weight

    def _load_stop_words_base(self):
        """
        加载停用词
        :return: 停用词集合
        """
        stop_words_base = set()

        try:
            model_dir_wildcard = os.path.join(_BASE_DIR, "model/AttackPatternDetectionModel_*")
            matched_paths = glob.glob(model_dir_wildcard)

            stop_words_base_path = os.path.realpath(
                os.path.join(matched_paths[0], KeywordsBaseConfig.STOP_WORDS_BASE_PATH)
            )
            with (os.fdopen(
                    os.open(stop_words_base_path, FilePermissions.FLAGS, FilePermissions.MODE), "r", encoding="utf-8"
            ) as stop_words_base_file):
                for line in stop_words_base_file:
                    stop_words_base.add(line.strip())
            self.logger.info(f"[AI Guardrail] agent pattern detection: The stop-word base "
                             f"loaded {len(stop_words_base)} lines.")
        except FileNotFoundError:
            self.logger.error("[AI Guardrail] agent pattern detection: The stop-word base not found.")
            raise
        except Exception as e:
            self.logger.error("[AI Guardrail] agent pattern detection: The stop-word base loaded failed.")
            raise

        return stop_words_base

    def _load_pattern_base(self):
        """
        加载攻击模式库
        :return: 攻击模式字典列表
        """
        pattern_base = []

        pattern_base_file = self._read_json_file(PatternBaseConfig.PATTERN_BASE_PATH)
        # 键idx，context，words
        for pattern_body in pattern_base_file:
            pattern_body["word_counts"] = defaultdict(int)
            pattern_body["keyword_counts"] = defaultdict(int)
            pattern_body["word_weights_map"] = defaultdict(int)
            for word in self.keywords_id:
                # 统计文档中各关键词的数量
                pattern_body["keyword_counts"][word] += 1
                # 统计文档中所有词的数量
                pattern_body["word_counts"][word] += 1
                # 统计文档中所有词的权重，非关键词默认为1
                pattern_body["word_weights_map"][word] = self.keywords_weight.get(word) \
                    if word in self.keywords_weight else 1

            # 原型中doc_hash
            doc_keyword_weights = [
                self.keywords_weight[word]
                if word in self.keywords_weight else 0 for word in pattern_body.get("words")
            ]
            pattern_body["doc_hash"] = self.hash_similarity_calculator.sentence_hash(pattern_body.get("words"),
                                                                                     doc_keyword_weights)

            pattern_base.append(pattern_body)
        self.logger.info(f"[AI Guardrail] agent pattern detection: The pattern base loaded {len(pattern_base)} lines.")

        return pattern_base

    def _load_bayes_prob_base(self):
        """
        加载贝叶斯概率信息
        """
        # 处理文件读取异常时，在exception中给keywords_bayes_prob_base返回默认值
        keywords_bayes_prob_base = self._read_json_file(KeywordsBaseConfig.KEYWORDS_BAYES_PROB_PATH)
        for word in keywords_bayes_prob_base:
            if word in self.keywords_id.keys():
                idx = self.keywords_id.get(word)
                self.keywords_base[idx]["bayes_prob"] = keywords_bayes_prob_base.get(word)

        self.logger.info(f"[AI Guardrail] agent pattern detection: The pattern base "
                         f"loaded {len(keywords_bayes_prob_base)} lines.")

        return keywords_bayes_prob_base


class KeywordsFeatures:
    """
    关键词库相关特征类
    """

    def __init__(self, param, logger):
        self.param = param
        self.logger = logger

    def filter_by_keywords_count(self, matched_keywords):
        """
        基于关键词数量进行文本过滤，如果文本中匹配到的关键词小于阈值，直接放行
        :param matched_keywords: 已匹配关键词列表
        :return: 是否超出阈值范围
        """
        matched_keywords_length = len(matched_keywords)
        if matched_keywords_length < self.param.get("keywords_config").get("KEYWORD_MATCH_THRESHOLD_NEG"):
            self.logger.info("[AI Guardrail] agent pattern detection: matched_keywords_length "
                             "is less than KEYWORD_MATCH_THRESHOLD and return Positive.")
            return KeywordsFeatureConfig.POSITIVE_FLAG
        elif matched_keywords_length > self.param.get("keywords_config").get("KEYWORD_MATCH_THRESHOLD_POS"):
            self.logger.info("[AI Guardrail] agent pattern detection: matched_keywords_length "
                             "is more than KEYWORD_MATCH_THRESHOLD and return Negative.")
            return KeywordsFeatureConfig.NEGATIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return matched_keywords_length doubt.")
            return KeywordsFeatureConfig.DOUBT_FLAG

    @staticmethod
    def _fetch_ngram_bayes_probs(matched_keywords, keywords_base_handler):
        """
        基于n-gram的贝叶斯分类器所需的数据，返回贝叶斯概率值以及下标
        :param matched_keywords: 文本分词后的词列表
        :param keywords_base_handler: 关键词库初始化对象实例
        :return: 权重列表，格式为(id, weight, word)，id指的是在words中第几个
        """
        weights = [
            (i, keywords_base_handler.keywords_bayes_prob_base.get(word), word)
            for (i, word) in enumerate(matched_keywords)
            if word in keywords_base_handler.keywords_bayes_prob_base
        ]

        return weights

    def _calc_ngram_probs(self, matched_keywords, matched_keywords_pos, keywords_base_handler):
        """
        计算关键词n_gram概率
        :param matched_keywords: 已匹配关键词列表
        :param matched_keywords_pos: 已匹配词在关键词库中的下标列表
        :param keywords_base_handler: 关键词库初始化对象实例
        :return: 返回已匹配关键词n_gram概率
        """
        bayes_probs = self._fetch_ngram_bayes_probs(matched_keywords, keywords_base_handler)
        bayes_probs_ngram = [
            (matched_keywords_pos[bayes_probs[i][0]], bayes_probs[i][1])
            for i in range(len(bayes_probs))
        ]

        pos = [bayes_probs_ngram[i][0] for i in range(len(bayes_probs_ngram))]
        weights = [bayes_probs_ngram[i][1] for i in range(len(bayes_probs_ngram))]

        pos_diff = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
        pos_diff_labels = [
            1 if pos_diff[i] < self.param.get("keywords_config").get("BAYES_NGRAM_MARGIN") else 0
            for i in range(len(pos_diff))
        ]
        pos_diff_labels.append(0)

        pos_diff_sum = [0] * len(pos_diff_labels)
        length_minus = len(pos_diff_labels) - self.param.get("keywords_config").get("BAYES_NGRAM")
        for i in range(length_minus):
            pos_diff_sum[i] = sum([
                pos_diff_labels[i + j] for j in range(self.param.get("keywords_config").get("BAYES_NGRAM"))
            ])

        ngram_bayes_prob = sum(
            [
                weights[i] for i in range(length_minus)
                if pos_diff_sum[i] == self.param.get("keywords_config").get("BAYES_NGRAM")
            ]
        )

        return ngram_bayes_prob

    def filter_by_bayes_prob(self, matched_keywords, matched_keywords_pos, keywords_base_handler):
        """
        基于关键词贝叶斯概率进行文本过滤
        :param matched_keywords: 已匹配关键词列表
        :param matched_keywords_pos: 已匹配词在关键词库中的下标列表
        :param keywords_base_handler: 关键词库初始化对象实例
        :return: 基于贝叶斯概率的判定标记，关键词n_gram概率特征
        """
        keywords_bayes_prob = self._calc_ngram_probs(matched_keywords, matched_keywords_pos,
                                                     keywords_base_handler)

        if keywords_bayes_prob < self.param.get("keywords_config").get("BAYES_PROB_THRESHOLD_NEG"):
            self.logger.info("[AI Guardrail] agent pattern detection: matched keywords "
                             "is less than BAYES_PROB_THRESHOLD and return Positive.")
            flag = KeywordsFeatureConfig.POSITIVE_FLAG
        elif keywords_bayes_prob > self.param.get("keywords_config").get("BAYES_PROB_THRESHOLD_POS"):
            self.logger.info("[AI Guardrail] agent pattern detection: matched keywords "
                             "is more than BAYES_PROB_THRESHOLD and return Negative.")
            flag = KeywordsFeatureConfig.NEGATIVE_FLAG
        else:
            self.logger.info("[AI Guardrail] agent pattern detection: return keywords_bayes_prob doubt.")
            flag = KeywordsFeatureConfig.DOUBT_FLAG

        return flag, keywords_bayes_prob

    def _calc_keywords_dist(self, matched_keywords_pos):
        """
        计算关键词距离
        :param matched_keywords_pos: 已匹配词在关键词库中的下标列表
        :return: 关键词距离特征
        """
        keyword_dists = []
        for i in range(1, len(matched_keywords_pos)):
            keyword_dists.append(matched_keywords_pos[i] - matched_keywords_pos[i - 1])
        keyword_dists = sorted(keyword_dists)

        keyword_dists_length = len(keyword_dists)
        qt_pos = [int(keyword_dists_length * rate) for rate in self.param.get("keywords_config").get("QT_POS_RATE")]
        dist_mean = float(sum(keyword_dists)) / keyword_dists_length
        dist_qt = [keyword_dists[pos] for pos in qt_pos]

        keyword_dist_feature = {"qt": dist_qt, "mean": dist_mean}

        self.logger.info("[AI Guardrail] agent pattern detection: get keyword_dist_feature successfully.")

        return keyword_dist_feature

    def filter_by_keywords_dist(self, matched_keywords_pos):
        """
        基于关键词距离进行文本过滤
        :param matched_keywords_pos: 已匹配词在关键词库中的下标列表
        :return: 基于关键词距离的判定标记，关键词距离特征
        """
        keyword_dist_feature = self._calc_keywords_dist(matched_keywords_pos)

        qts = [keyword_dist_feature.get("qt")[i] for i in self.param.get("keywords_config").get("QT_CHECK_POS")]

        flag = KeywordsFeatureConfig.POSITIVE_FLAG
        for (i, pos_i) in enumerate(self.param.get("keywords_config").get("QT_CHECK_POS")):
            if qts[pos_i] < self.param.get("keywords_config").get("QT_CHECK_THRESHOLD")[i]:
                flag = KeywordsFeatureConfig.DOUBT_FLAG
                self.logger.info("[AI Guardrail] agent pattern detection: return keyword_dist_feature doubt.")
                break

        if flag == KeywordsFeatureConfig.POSITIVE_FLAG:
            self.logger.info("[AI Guardrail] agent pattern detection: matched keywords "
                             "is more than QT_CHECK_THRESHOLD and return Positive.")

        return flag, keyword_dist_feature
