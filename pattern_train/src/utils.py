#!/usr/bin/env python
# coding: utf-8
"""
功能描述: 字符串处理工具
"""

import re
import string

import jieba

PUNC_PATTERN = r"[\u2000-\u206f\u3000-\u303f\uff00-\uffef]"


def load_stop_words(stopword_filename: str):
    """
    加载停顿词库
    :param stopword_filename: 停顿词库的绝对路径
    :return: 停顿词库集合
    """
    stop_words = set()
    with open(stopword_filename, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    return stop_words


def judge_end_with_punc(s: str):
    """
    判断文本结尾最后一个字符是不是符号
    :param s: 输入文本
    :return: True表示是符号
    """
    if len(re.findall(PUNC_PATTERN, s[-1:])) > 0:
        return True
    if s[-1:] in string.punctuation:
        return True
    return False


def cut_words(s: str):
    """
    通过jieba分词，在分词后会检查每个单词是否为符号或是否包含符号，如果是则删除
    :param s: 分词前文本
    :return: 分词后的结果
    """
    segs = jieba.cut(s)
    results = []
    for seg in segs:
        if seg not in string.punctuation and len(re.findall(PUNC_PATTERN, seg)) == 0:
            results.append(seg)
    return results


def cut_words_with_stop_words(s: str, stop_words: set[str]):
    """
    通过jieba分词，分词后不仅会检查是否为符号，还会检查是否在停顿词列表中
    :param s: 待分词文本
    :param stop_words: 停顿词集合
    :return: 分词后的结果
    """
    segs, results = cut_words(s), []
    for seg in segs:
        if seg not in stop_words:
            results.append(seg)

    return results


def remove_stop_words(s: str, stop_words: set[str]):
    """
    删除文本中的停顿词，输入为分词前的文本，输出为分词结果和整段新文本
    :param s: 分词前文本
    :param stop_words: 停顿词集合
    :return:  分词结果，删除停顿词后的新文本
    """
    segs = cut_words_with_stop_words(s, stop_words)
    return segs, ''.join(segs)
