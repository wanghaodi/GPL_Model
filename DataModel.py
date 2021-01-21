#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/1/21 5:47 下午
# @Author  : Haodi Wang
# @FileName: DataModel.py
# @Software: PyCharm
# @contact: whdi@foxmail.com
#           whd@seu.edu.cn

import torch
import dgl
import pytorch_lightning as pl
from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader, random_split

def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

#  本类用于数据加载
class MyDataModule(pl.LightningDataModule):
    def __init__(self, trainset, batch_size=512):
        super().__init__()
        self.batch_size = batch_size
        self.trainset = trainset

    def setup(self, stage):
        # # 实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        if stage == 'fit' or stage is None:
            # trainset = MiniGCDataset(20000, 10, 20)
            train_dataset, val_dataset = random_split(self.trainset, [14000, 6000])
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == 'test' or stage is None:
            self.test_dataset = MiniGCDataset(10000, 10, 20)

    def prepare_data(self):
        # 一般是从网络中下载数据
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10,
                          collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10,
                          collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=collate)

    # def transfer_batch_to_device(self, batch, device):
    #     x = batch['x']
    #     x = CustomDataWrapper(x)
    #     batch['x'].to(device)
    #     return batch