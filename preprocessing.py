# -*- coding: utf-8 -*-
'''
preprocessing of word cut training and test dataset, labeling method uses 4-tag B(begining),M(median),
E(ending),S(single)
'''

import os
import codecs
from typing import List,Set,Dict

def labeling_corpus(corpus:List[List[str]])->List[List[str]]:
    res = []
    for sentence in corpus:
        res.append(labeling_sent(sentence))
    return res

def labeling_sent(sent:List[str])->List[str]:
    res = []
    for words in sent:
        if len(words) == 1:
            res.append(words + '/' + 'S')
        elif len(words) == 2:
            res.append(words[0] + '/' + 'B')
            res.append(words[1] + '/' + 'E')
        else:
            res.append(words[0] + '/' + 'B')
            res.extend([w + '/' + 'M' for w in words[1:-1]])
            res.append(words[-1] + '/' + 'E')
    return res


def get_class(char):
    zh_num = [u'零', u'○', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'百', u'千', u'万']
    ar_num = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'.', u'０', u'１', u'２', u'３', u'４', u'５',
              u'６', u'７', u'８', u'９']
    date = [u'日', u'年', u'月']
    letter = ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    if char in zh_num or char in ar_num:
        return '1'
    elif char in date:
        return '2'
    elif char in letter:
        return '3'
    else:
        return '4'

def isPu(char):
    punctuation = [u'，', u'。', u'？', u'！', u'；', u'－－', u'、', u'——', u'（', u'）', u'《', u'》', u'：', u'“', u'”', u'’',
                   u'‘']
    if char in punctuation:
        return '1'
    else:
        return '0'


def feature_extraction_test(corpus:List[List[str]]):
    features = []
    for sent in corpus:
        for i,word in enumerate(sent[:-2],start=0):
            curr_char = word[0]
            pre_pre_char = sent[i - 2] if i - 2 >= 0 else '_'
            pre_char = sent[i - 1] if i - 1 >= 0 else '_'
            next_char = sent[i + 1]
            next_next_char = sent[i + 2]
            features.append(
                ' '.join(
                    [
                        'C-2='+pre_pre_char,
                        'C-1='+pre_char,
                        'C0=' + curr_char,
                        'C1=' + next_char,
                        'C2=' + next_next_char,
                        'C-2='+pre_pre_char+'C-1='+pre_char,
                        'C-1='+pre_char+'C0='+curr_char,
                        'C0='+curr_char+'C1='+next_char,
                        'C1='+next_char+'C2='+next_next_char,
                        'C-1='+pre_char+'C1='+next_char,
                        'C-2=' + pre_pre_char + 'C-1=' + pre_char+'C0='+curr_char,
                        'C-1=' + pre_char + 'C0=' + curr_char+'C1='+next_char,
                        'C0=' + curr_char + 'C1=' + next_char+'C2='+next_next_char,
                        'Pu='+isPu(curr_char),
                        'Tc-2='+get_class(pre_pre_char)+'Tc-1='+get_class(pre_char),
                        'Tc0='+get_class(curr_char)+'Tc1='+get_class(next_char),
                        'Tc2='+get_class(next_next_char)
                    ]
                )
            )
    return features


def feature_extraction_train(corpus:List[List[str]]):
    features = []
    for sent in corpus:
        for i,word in enumerate(sent[:-2],start=0):
            tag = word[2]
            curr_char = word[0]
            pre_pre_char = sent[i-2][0] if i - 2>=0 else '_'
            pre_char = sent[i-1][0] if i - 1>=0 else '_'
            next_char = sent[i+1][0]
            next_next_char = sent[i+2][0]
            features.append(
                ' '.join(
                    [
                        tag,
                        'C-2='+pre_pre_char,
                        'C-1='+pre_char,
                        'C0=' + curr_char,
                        'C1=' + next_char,
                        'C2=' + next_next_char,
                        'C-2='+pre_pre_char+'C-1='+pre_char,
                        'C-1='+pre_char+'C0='+curr_char,
                        'C0='+curr_char+'C1='+next_char,
                        'C1='+next_char+'C2='+next_next_char,
                        'C-1='+pre_char+'C1='+next_char,
                        'C-2=' + pre_pre_char + 'C-1=' + pre_char+'C0='+curr_char,
                        'C-1=' + pre_char + 'C0=' + curr_char+'C1='+next_char,
                        'C0=' + curr_char + 'C1=' + next_char+'C2='+next_next_char,
                        'Pu='+isPu(curr_char),
                        'Tc-2='+get_class(pre_pre_char)+'Tc-1='+get_class(pre_char),
                        'Tc0='+get_class(curr_char)+'Tc1='+get_class(next_char),
                        'Tc2='+get_class(next_next_char)
                    ]
                )
            )
    return features

def read_lines(path):
    content = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            content.append(line.strip())
    return content

def write_lines(path,content):
    with open(path, 'w', encoding='utf8') as f:
        f.write('\n'.join(content))


def prepare_data(train_path,test_path):
    train_corpus = [line.split() for line in read_lines(train_path)]
    test_corpus = [list(line) for line in read_lines(test_path)]

    train_corpus_label = labeling_corpus(train_corpus)
    train_features = feature_extraction_train(train_corpus)
    test_features = feature_extraction_test(test_corpus)
    write_lines('./data/train_features.txt',train_features)
    write_lines('./data/test_features.txt',test_features)
