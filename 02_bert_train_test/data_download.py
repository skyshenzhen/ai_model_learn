# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         data_download.py
# Description:  load dataset from huggingface datasets and to csv
# Author:       shaver
# Date:         2025/5/8
# -------------------------------------------------------------------------------


from datasets import load_dataset
# 在线加载数据
dataset = load_dataset(path="lansinuote/ChnSentiCorp", cache_dir="data/")

train_data = dataset["train"]
for data in train_data:
    print(data)
# 转为csv格式
train_data.to_csv("data/train.csv", index=False)
