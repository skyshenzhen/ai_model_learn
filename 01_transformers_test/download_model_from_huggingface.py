# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         download_model_from_huggingface.py
# Description:  download model and tokenizer from huggingface
# Author:       shaver
# Date:         2025/05/07
# -------------------------------------------------------------------------------


# 将模型下载到本地调用
from transformers import AutoModelForCausalLM, AutoTokenizer

# 将模型和分词工具下载到本地，并指定保存路径
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "/Users/shaver/PycharmProjects/ai_model_learn/01_transformers_test/model/uer/gpt2-chinese-cluecorpussmall"

# 下载模型
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# 下载分词工具
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已下载到：{cache_dir}")
