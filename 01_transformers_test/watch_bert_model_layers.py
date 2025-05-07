# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         watch_bert_model_layers.py
# Description:  watch bert model layers
# Author:       shaver
# Date:         2025/05/07
# -------------------------------------------------------------------------------

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# 加载模型和分词器
model_name = r"/Users/shaver/Downloads/2.资料/大模型学习/第四期/1 day01_HuggingFace核心组件介绍/trasnFormers_test/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建分类 pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,device="cpu")

# 进行分类
result = classifier("你好，我是一款语言模型")
print(result)
print(model)


