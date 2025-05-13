# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         data.py
# Description:  This is a script for loading data.
# Author:       shaver
# Date:         2025/5/12
# -------------------------------------------------------------------------------

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        with open("data/chinese_poems.txt", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [i.strip() for i in lines]
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.lines[item]


if __name__ == '__main__':
    dataset = MyDataset()
    for data in dataset:
        print(data)
