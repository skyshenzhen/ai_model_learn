# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         llms
# Description:  大模型
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
from crewai import LLM

def deepseek_llm():
    return LLM(
        model="deepseek-chat",  # or whatever the correct model name is for DeepSeek
        temperature=0.7,
        base_url="https://api.deepseek.com/v1",  # modified to DeepSeek's API endpoint
        api_key="sk-aa3a24ed00744632943daf833f298803"  # make sure this is your DeepSeek API key
    )