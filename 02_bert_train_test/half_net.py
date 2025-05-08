# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         half_net.py
# Description:  定义增量模型，冻结Bert模型参数，只训练增量模型，实现二分类任务
# Author:       shaver
# Date:         2025/5/8
# -------------------------------------------------------------------------------

import torch
from transformers import BertModel

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# 加载预训练模型
pretrained = BertModel.from_pretrained(
    r"model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(
    DEVICE)
print(pretrained)


# 定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768, 2)

    # 使用模型处理数据（执行前向计算）
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        return out
