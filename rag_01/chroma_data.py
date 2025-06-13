# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         chroma_data
# Description:
# Author:       shaver
# Date:         2025/6/13
# -------------------------------------------------------------------------------
import chromadb
from chromadb.utils import embedding_functions


# 数据保存至本地目录

# 默认情况下，Chroma 使用 DefaultEmbeddingFunction，它是基于 Sentence Transformers 的 MiniLM-L6-v2 模型
default_ef = embedding_functions.DefaultEmbeddingFunction()
#default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="/Users/shaver/PycharmProjects/ai_model_learn/rag_01/save")

def create_collection(name:str):
    """
    :param name: collection 名称
    :return: 返回
    """
    collection = client.create_collection(
        name = name,
        configuration = {
            # HNSW 索引算法，基于图的近似最近邻搜索算法（Approximate Nearest Neighbor，ANN）
            "hnsw": {
                "space": "cosine", # 指定余弦相似度计算
                "ef_search": 100,
                "ef_construction": 100,
                "max_neighbors": 16,
                "num_threads": 4
            },
            # 指定向量模型
            "embedding_function": default_ef
        }
    )
    return collection

def get_collection(name:str):
    """
    :param name: collection 名称
    :return: 返回 collection
    """
    collection = client.get_collection(name)
    return collection

def modify_collection_name(name:str, new_name:str):
    """
    :param name: collection 名称
    :param new_name: 新的 collection 名称
    :return: 返回
    """
    collection = get_collection(name)
    collection.modify(name=new_name)

def delete_collection(name:str):
    """
    :param name: collection 名称
    :return:
    """
    client.delete_collection(name)





def insert_collection_data(collection, data):
    """

    :param collection:
    :param data:
    :return:
    """
    # 方式1：自动生成向量（使用集合指定的嵌入模型）
    collection.add(
        documents=data['documents'],
        metadatas=data['metadatas'],
        ids=data['ids']
    )

    # 方式2：手动传入预计算向量
    # collection.add(
    #     embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    #     documents = ["文本1", "文本2"],
    #     ids = ["id3", "id4"]
    # )

def search_collection_data(collection, query_texts, top_k=10):
    """
    :param collection: collection 名称
    :param query_texts:  查询文本
    :param top_k:  返回结果数
    :return:  返回查询结果
    """
    results = collection.query(
        query_texts=query_texts,
        n_results=top_k,
        # where = {"source": "RAG"}, # 按元数据过滤
        # where_document = {"$contains": "检索增强生成"} # 按文档内容过滤
    )
    return results

def modify_collection_data(collection, data):
    """

    :param collection:  collection 名称
    :param data:  数据
    :return:
    """
    collection.update(
        ids=data.ids,
        documents=data.documents
    )

def delete_collection_data(collection, ids):
    """

    :param collection: collection 名称
    :param ids:  删除ids
    :return:
    """
    collection.delete(ids=ids)

if __name__ == '__main__':
    client = chromadb.PersistentClient(path="/Users/shaver/PycharmProjects/ai_model_learn/rag_01/save")
    collection = client.get_collection(name="test_collection")
    # 方式1：自动生成向量（使用集合指定的嵌入模型）
    collection.add(
        documents=["RAG是一种检索增强生成技术", "向量数据库存储文档的嵌入表示",
                   "在机器学习领域，智能体（Agent）通常指能够感知环境、做出决策并采取行动以实现特定目标的实体"],
        metadatas=[{"source": "RAG"}, {"source": "向量数据库"}, {"source": "Agent"}],
        ids=["id1", "id2", "id3"]
    )

    print(11)