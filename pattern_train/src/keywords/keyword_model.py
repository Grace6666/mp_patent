#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 关键词模型
修改时间: 2025/8/1 10:00
"""

from keywords.ac import AC


class KeywordModel:

    # 模型默认超参数
    config: dict = {
        'qt_pos_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'qt_check_pos': [0, 1],
        'qt_check_threshold': [20, 30],
        'keyword_match_threshold': {'neg': 2, 'pos': 40},
        'bayes_prob_threshold': {'neg': 0.5, 'pos': 0.9},
        'bayes_ngram': 3,
        'bayes_ngram_margin': 3
    }

    def __init__(self, ac: AC, config: dict = None):
        """
        初始化关键词特征模型
        :param ac: AC自动机模型
        :param config: 超参配置
        """
        if config is not None:
            self.config = config
        self.ac = ac

    @staticmethod
    def calc_bayes_probs(bayes_probs: list[float]):
        """
        计算页面整体的贝叶斯概率
        :param bayes_probs: 页面关键词的贝叶斯概率值，跟数据库返回的格式一致
        :return: 返回整个页面的整体贝叶斯概率值
        """
        s, count = 0, 0
        for p in bayes_probs:
            s += p
            count += 1
        return s

    def keyword_match(self, page: str):
        """
        关键词匹配
        :param page: 页面字符串表示（无需分词）
        :return: 匹配结果列表，（在页面中的位置，匹配到第几个关键词，关键词文本）
        """
        # matched_result：元组，第一个是匹配的模式 第二个元素是匹配模式起始模式、结束位置
        # match_result：元组 --- 匹配关键词的pos  匹配关键词的id  匹配关键词
        matched_results = self.ac.match_results(page)
        # 拆分matched_result,将匹配的关键词的 （pos id 关键词）分别构造列表
        return [x[0] for x in matched_results], [x[1] for x in matched_results], [x[2] for x in matched_results]

    def keyword_match_(self, page: str):
        '''
        关键词匹配精简输出，只保留在页面中的位置和关键词文本
        :param page: 页面字符串表示（无需分词）
        :return: 匹配结果列表，（在页面中的位置，关键词文本）
        '''
        matched_results = self.ac.match_results(page)
        matched_keywords = [(x[0], x[2]) for x in matched_results]
        return matched_keywords

    def judge_by_keyword_match(self, keyword_feature: dict):
        """
        关键词匹配数量判定，如果界面中匹配到的关键词小于某个阈值，则直接放行
        :param keyword_feature: 匹配到的关键词特征
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if 'keyword_match' not in keyword_feature:
            return 0
        if keyword_feature['keyword_match'] < self.config['keyword_match_threshold']['neg']:
            return -1
        if keyword_feature['keyword_match'] > self.config['keyword_match_threshold']['pos']:
            return 1

        return 0

    def keyword_dists_feature(self, matched_keywords_pos: list[int]):
        """
        构造关键词间距离特征
        :param matched_keywords_pos: 页面匹配到关键词的位置下标
        :return: 关键词间距离特征集合
        """
        keyword_dists = []
        for i in range(1, len(matched_keywords_pos)):
            keyword_dists.append(matched_keywords_pos[i] - matched_keywords_pos[i - 1])
        keyword_dists = sorted(keyword_dists)

        qt_pos = []
        for rate in self.config['qt_pos_rate']:
            qt_pos.append(int(len(keyword_dists) * rate))

        dist_mean = 0.0
        for d in keyword_dists:
            dist_mean += d
        dist_mean /= len(keyword_dists)

        dist_qt = []
        for pos in qt_pos:
            dist_qt.append(keyword_dists[pos])

        features = {
            'qt': dist_qt,
            'mean': dist_mean
        }

        return features

    def judge_by_keyword_dists(self, keyword_feature):
        """
        关键词间距离判定，如果界面中关键词之间的平均距离分位值都大于特定的阈值，则直接放行
        :param keyword_feature: 关键词特征集合
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if 'keyword_dist_feature' not in keyword_feature:
            return 0

        keyword_dist_feature = keyword_feature['keyword_dist_feature']
        qts = [keyword_dist_feature['qt'][idx] for idx in self.config['qt_check_pos']]
        for i in range(len(self.config['qt_check_pos'])):
            if qts[self.config['qt_check_pos'][i]] < self.config['qt_check_threshold'][i]:
                return 0

        return -1

    def calc_ngram_probs(self, bayes_probs_ngram: list[int, float]):
        margin = self.config['bayes_ngram_margin']
        gram = self.config['bayes_ngram']

        pos = [bayes_probs_ngram[i][0] for i in range(len(bayes_probs_ngram))]
        weights = [bayes_probs_ngram[i][1] for i in range(len(bayes_probs_ngram))]

        diffs_ = [pos[i+1]-pos[i] for i in range(len(pos)-1)]
        diffs = [0] * len(diffs_)
        for i, diff in enumerate(diffs_):
            if diff < margin:
                diffs[i] = 1
        diffs.append(0)
        sums = [0] * len(diffs)
        for i in range(len(diffs)-gram):
            s = 0
            for j in range(gram):
                s += diffs[i+j]
            sums[i] = s

        bayes_prob = 0
        for i in range(len(diffs)-gram):
            if sums[i] == gram:
                bayes_prob += weights[i]

        return bayes_prob

    def judge_by_bayes_probs(self, keyword_feature: dict):
        """
        通过贝叶斯概率值判断是否放行
        :param keyword_feature: keyword_feature['bayes_prob']整个页面的贝叶斯概率值
        :return: 是否放行，1表示检出攻击，-1表示放行，0表示进入下一个环节
        """
        if keyword_feature['bayes_prob'] < self.config['bayes_prob_threshold']['neg']:
            return -1
        if keyword_feature['bayes_prob'] > self.config['bayes_prob_threshold']['pos']:
            return 1
        return 0
