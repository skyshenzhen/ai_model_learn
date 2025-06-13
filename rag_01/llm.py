# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         llm
# Description:  定义AI大模型
# Author:       shaver
# Date:         2025/6/13
# -------------------------------------------------------------------------------
import os
from openai import OpenAI

# 需要在系统环境变量中配置好相应的key
# OpenAI key（1 代理方式 2 官网注册购买）

# 阿里百炼配置
# DASHSCOPE_API_KEY sk-xxx
# DASHSCOPE_BASE_URL https://dashscope.aliyuncs.com/compatible-mode/v1
client = OpenAI(
    # api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    api_key="sk-e7271b80bcce47f6ae33ae7c545e8cef",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
