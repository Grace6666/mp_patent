#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 快筛模块特征提取类
"""

from keywords.ac import AC
from utils import cut_words
from db.keyword_db import KeywordDB
from fusion.fusion_model import FuseModel
from keywords.keyword_model import KeywordModel
from doc.doc_model import DocModel, sentence_hash
from utils import remove_stop_words, load_stop_words


class FastDetectorControllerFeature:
    """
    功能：特征提取类，用于从输入页面中提取关键词、文档特征和融合特征
    这些特征可以用于训练模型或进行模型推断
    """
    # 超参信息
    config: dict = {}

    # AC自动机模型
    ac: AC = None

    # 关键词数据库
    db: KeywordDB = None

    # 关键词特征模型
    keyword_model: KeywordModel = None

    # 文档特征模型
    doc_model: DocModel = None

    # 融合特征模型
    fuse_model: FuseModel = None

    # 停顿词列表
    stop_words: set = None

    def __init__(self, config: dict = None):
        if config is not None:
            self.config = config

        self.init_modules()

    def init_modules(self):
        """
        初始化整个判定模块
        """
        # 初始化关键词数据库
        self.db = KeywordDB(self.config['db_config']['keyword'])
        self.db.load_from_file(self.config['db_config']['keyword_dir'])
        self.db.load_bayes_prob(self.config['db_config']['keyword_dir'])

        # jieba初始化需要0.5s，在这里先做初始化
        cut_words('测试分词功能。Test cut words!!')

        # 获取数据库中所有关键词的字符串，返回词列表keywords
        keywords = self.db.fetch_all_keywords()

        # 加载停顿词库，返回停顿词库集合stop_words-----set
        self.stop_words = load_stop_words(self.config['data_config']['stopwords_filename'])

        # 初始化关键词特征、文档、融合评估模型模型
        self.keyword_model = KeywordModel(ac=AC(keywords), config=self.config['feature_config']['keyword'])
        # 初始化文档模型
        self.doc_model = DocModel(config=self.config['feature_config']['doc'])
        # 初始化融合评估模型
        self.fuse_model = FuseModel(config=self.config['feature_config']['fuse'])

        # 返回文档分词后的列表
        docs = self.db.fetch_all_docs_words()
        docs_hash = []
        for idx, words in docs:
            # 提取每个文档中的关键词权重，返回列表
            doc_keyword_weights = self.db.fetch_page_keyword_weights(words)
            docs_hash.append((idx, sentence_hash(words, doc_keyword_weights)))
        # 文档hash值存储会数据库
        self.db.set_docs_hash(docs_hash)

    def make_features(self, page: str, filter_switch=False):
        """
        特征构造函数，用于融合判定模型的训练，以及贝叶斯概率的建模
        :param page: 页面字符串（不需要分词）
        :param filter_switch: 特征过滤开关，获取部分特征时（如小艺claw注入攻击检测）开启该开关
        :return: 页面特征列表
        """
        # 移除停顿词
        segs, page = remove_stop_words(page, self.stop_words)

        # 关键词特征
        keyword_feature = {}
        # 文本特征
        doc_feature = {}

        # Step 1: 匹配界面中的关键词，并做第一次的筛选，返回匹配关键词的在文本中的位置 id 关键词 三个列表
        matched_keywords_pos, matched_keywords_idx, matched_keywords = self.keyword_model.keyword_match(page)
        # 匹配关键词数作为特征
        keyword_feature['keyword_match'] = len(matched_keywords)

        # 关键词匹配数量判定，如果界面中匹配到的关键词小于某个阈值，则直接放行
        if self.keyword_model.judge_by_keyword_match(keyword_feature) != 0:
            return segs, {'keyword_feature': keyword_feature,
                          'keywords': list(zip(matched_keywords_pos, matched_keywords))}

        # Step 2: 计算贝叶斯分类器的概率
        bayes_probs = self.db.fetch_page_keyword_bayes_probs_ngram(matched_keywords)
        bayes_probs_ngram = [(matched_keywords_pos[bayes_prob[0]], bayes_prob[1]) for bayes_prob in bayes_probs]
        bayes_prob = self.keyword_model.calc_ngram_probs(bayes_probs_ngram)
        keyword_feature['bayes_prob'] = bayes_prob

        # Step 3: 计算界面关键词的特征，并做第二次的筛选
        keyword_dist_features = self.keyword_model.keyword_dists_feature(matched_keywords_pos)
        keyword_feature['dist_feature'] = keyword_dist_features

        matched_docs = []
        # 如果filter_switch默认为False，继续进行Step 4和Step 5获取如下特征
        if not filter_switch:
            # Step 4: 获取跟界面相关的攻击手法，做第三次的筛选
            inverse_ids = self.db.fetch_inverse_doc_ids(matched_keywords_idx)
            doc_ids = self.doc_model.jaccard_filter(inverse_ids)
            doc_feature['doc_match'] = len(doc_ids)


            if len(doc_ids) > 0:
                matched_docs_ = self.db.fetch_docs_words(doc_ids)
                for words in matched_docs_:
                    matched_docs.append(''.join(words))
            if self.doc_model.judge_by_inverse_docs(doc_feature) != 0:
                return segs, {
                    'keyword_feature': keyword_feature,
                    'keywords': list(zip(matched_keywords_pos, matched_keywords)),
                    'doc_feature': doc_feature,
                    'docs': matched_docs
                }

            # Step 5：计算文本相似度匹配特征，做第四次筛选
            page_words = cut_words(page)
            page_keywords_weights = self.db.fetch_page_keyword_weights(page_words)

            docs_info_for_hash_sim = self.db.fetch_docs_for_hash_sim(doc_ids)
            docs_info_for_weight_sim = self.db.fetch_docs_for_weight_sim(doc_ids)

            hash_sims = self.doc_model.calc_hash_sims(page_words, page_keywords_weights, docs_info_for_hash_sim)
            weight_sims = self.doc_model.calc_weight_sims(page_words, docs_info_for_weight_sim)

            doc_feature['hash_sim'] = hash_sims
            doc_feature['weight_sim'] = weight_sims

        # 交给决策树做最后的融合判断
        return segs, {
            'keyword_feature': keyword_feature,
            'keywords': list(zip(matched_keywords_pos, matched_keywords)),
            'doc_feature': doc_feature,
            'docs': matched_docs
        }

    def make_fuse_features(self, feature, filter_switch=False):
        doc_feature = {}
        try:
            keyword_feature = feature['keyword_feature']
            if filter_switch:
                seg_feature_dict = feature["seg_feature"]
                seg_feature = list(seg_feature_dict.values())
                fuse_feature = self.fuse_model.fuse_seg_and_key_features(seg_feature, keyword_feature)
            else:
                doc_feature = feature['doc_feature']
                fuse_feature = self.fuse_model.fuse_features(keyword_feature, doc_feature)
        except Exception as e:
            return None

        if self.keyword_model.judge_by_keyword_match(keyword_feature) != 0:
            return None

        if self.keyword_model.judge_by_bayes_probs(keyword_feature) != 0:
            return None

        if self.keyword_model.judge_by_keyword_dists(keyword_feature) != 0:
            return None

        if self.doc_model.judge_by_inverse_docs(doc_feature) != 0:
            return None

        if self.doc_model.judge_by_hash_sim(doc_feature) != 0:
            return None

        if self.doc_model.judge_by_weight_sims(doc_feature) != 0:
            return None

        return fuse_feature
