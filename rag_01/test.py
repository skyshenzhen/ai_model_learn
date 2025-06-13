# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         test
# Description:
# Author:       shaver
# Date:         2025/6/13
# -------------------------------------------------------------------------------
from rag_01.tool import get_embeddings, cos_sim, l2


def test01():
    """测试返回embedding向量"""
    test_query = ["hello world", "你好，世界"]
    print(f"Embedding length: {len(get_embeddings(test_query))}")
    vec = get_embeddings(test_query)[0]
    print(f"first vector Total dimension: {len(vec)}")
    print(f"First 10 elements: {vec[:10]}")


def test02():
    """测试余弦相似度 欧式距离"""
    query = "国际争端"

    # 且能支持跨语言
    # query = "global conflicts"
    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]

    query_vec = get_embeddings([query])[0]
    doc_vecs = get_embeddings(documents)

    print("Query与自己的余弦距离: {:.2f}".format(cos_sim(query_vec, query_vec)))
    print("Query与Documents的余弦距离:")
    for vec in doc_vecs:
        print(cos_sim(query_vec, vec))

    print()

    print("Query与自己的欧氏距离: {:.2f}".format(l2(query_vec, query_vec)))
    print("Query与Documents的欧氏距离:")
    for vec in doc_vecs:
        print(l2(query_vec, vec))


if __name__ == '__main__':
    test01()
