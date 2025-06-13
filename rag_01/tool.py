# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         tool
# Description:  定义工具 余弦相似度比较 欧式距离比较
# Author:       shaver
# Date:         2025/6/13
# -------------------------------------------------------------------------------

import numpy as np
from numpy import dot
from numpy.linalg import norm

from rag_01.llm import client


def cos_sim(a, b):
    """余弦距离 -- 越大越相似"""
    return dot(a, b) / (norm(a) * norm(b))


def l2(a, b):
    """欧氏距离 -- 越小越相似"""
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


def get_embeddings(texts, model="text-embedding-v1", dimensions=None):
    """获取embeddings"""
    if model == "text-embedding-v1":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(
            input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]
