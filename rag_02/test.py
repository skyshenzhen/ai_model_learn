# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         test
# Description:
# Author:       shaver
# Date:         2025/6/17
# -------------------------------------------------------------------------------
import os
import logging
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback

from rag_02.tool import PdfTool

if __name__ == '__main__':
    print("This is a test file.")
    load_dotenv()  # 加载 .env 文件中的变量


    # # 读取PDF文件
    # pdf_reader = PdfReader('./doc/11 大模型部署.pdf')
    # # 提取文本和页码信息
    # text, page_numbers = PdfTool.extract_text_with_page_numbers(pdf_reader)
    #
    # print(f"提取的文本长度: {len(text)} 个字符, 总行数: {len(page_numbers)} 行")
    #
    # # 处理文本并创建知识库，同时保存到磁盘
    # save_dir = "./vector_db"
    # knowledgeBase = PdfTool.process_text_with_splitter(text, page_numbers, save_path=save_dir)

    knowledgeBase = PdfTool.load_knowledge_base("./vector_db")
    # 设置查询问题
    # query = "客户经理被投诉了，投诉一次扣多少分"
    query = "大模型特点？"
    if query:
        # 执行相似度搜索，找到与查询相关的文档
        docs = knowledgeBase.similarity_search(query)

        # 初始化对话大模型
        chatLLM = ChatOpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="deepseek-v3"
        )

        # 加载问答链
        chain = load_qa_chain(chatLLM, chain_type="stuff")

        # 准备输入数据
        input_data = {"input_documents": docs, "question": query}

        # 使用回调函数跟踪API调用成本
        with get_openai_callback() as cost:
            # 执行问答链
            response = chain.invoke(input=input_data)
            print(f"查询已处理。成本: {cost}")
            print(response["output_text"])
            print("来源:")

        # 记录唯一的页码
        unique_pages = set()

        # 显示每个文档块的来源页码
        for doc in docs:
            text_content = getattr(doc, "page_content", "")
            source_page = knowledgeBase.page_info.get(
                text_content.strip(), "未知"
            )

            if source_page not in unique_pages:
                unique_pages.add(source_page)
                print(f"文本块页码: {source_page}")
