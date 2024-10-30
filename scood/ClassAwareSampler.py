"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb

##################################
## Class-aware sampling, partly implemented by from beijingwithlove
##################################

# 类感知采样，作用是循环遍历给定的数据集，并在每次完整的遍历之后随机打乱数据顺序，以便于下一次迭代
class RandomCycleIter:
    # 初始化函数
    def __init__ (self, data, test_mode=False):
        # 将数据转换为列表
        self.data_list = list(data)
        # 获取数据长度
        self.length = len(self.data_list)
        # 初始化索引
        self.i = self.length - 1
        # 设置测试模式
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        # 下一个元素
        self.i += 1
        
        # 如果索引超出范围，则重置索引并随机打乱数据
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]

# 类感知样本生成器，其目的是确保模型训练时从每个类别中均匀地抽取样本
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    # n表示采样的总次数
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        # 每num_samples_cls次更新为0
        if j >= num_samples_cls:
            j = 0

        # 生成样本
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

# 类感知采样器
class ClassAwareSampler (Sampler):
    
    def __init__(self, data_source, num_samples_cls=1,):
        # 计算类别的数量，默认为1000
        num_classes = len(np.unique(data_source.pseudo_label_pred))
        # 创建类迭代器
        self.class_iter = RandomCycleIter(range(num_classes))
        # 为每个类别创建一个列表
        cls_data_list = [list() for _ in range(num_classes)]
        # 获取每个类别的样本索引列表
        for i, label in enumerate(data_source.pseudo_label_pred):
            cls_data_list[label].append(i)
        # 获取每类样本的迭代器列表
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        # 获取采样的总频率，数量为类别数乘以最大的类内样本数
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls
        
    # 获取迭代器
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    # 获取迭代的总数量
    def __len__ (self):
        return self.num_samples

# 获取采样器函数
def get_sampler():
    return ClassAwareSampler

##################################