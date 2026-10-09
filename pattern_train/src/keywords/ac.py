#!/usr/bin/env python
# coding: utf-8
"""
功能描述: AC处理机
修改时间: 2025/8/1 10:00
"""

import ahocorasick


class AC:
    def __init__(self, keywords: list[str] = None):
        """
        模型初始化
        :param keywords: 输入为关键词文本列表，每个关键词的字符串
        """
        self.ac_tree = ahocorasick.Automaton()
        self.keywords = keywords
        for flag, word in enumerate(keywords):
            self.ac_tree.add_word(word, (flag, word))
        self.ac_tree.make_automaton()

    def match_results(self, sentence: str):
        """
        关键词匹配
        :param sentence: 页面的字符串（无需分词）
        :return: 匹配结果列表，（在页面中的位置，匹配到第几个关键词，关键词文本）
        """
        results = []
        # 查找所有匹配的模式
        for match in self.ac_tree.iter(sentence):
            results.append((match[0], match[1][0], match[1][1]))
        return results
