"""Adapted from: https://github.com/facebookresearch/odin"""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor

# ODIN评价
class SoftMaxpostprocessor(BasePostprocessor):
    def __init__(self, temperature: float = 1000.0, noise: float = 0.0014):
        super().__init__()

        self.temperature = temperature
        self.noise = noise

    def __call__(
        self,
        net: nn.Module,
        data: Any,
    ):
        data.requires_grad = True
        # 图像送入网络得到输出
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        # 初始化交叉熵损失
        criterion = nn.CrossEntropyLoss()
        
        # 获取对应的标签
        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        # 使用温度缩放
        output = output / self.temperature

        # 计算损失值
        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        # 归一化梯度到二值
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        # 从原始的编码中缩放值
        gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)

        # Adding small perturbations to images
        # 向图像添加小的扰动
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        # 获取网络的输出
        output = net(tempInputs)
        # 计算对应的softmax分数
        score = torch.softmax(output, dim=1)
        # 计算最大的softmax分数
        msp, _ = torch.max(score, dim=1)
        # 进行温度缩放
        output = output / self.temperature
        # Calculating the confidence after adding perturbations
        # 计算添加扰动后的信心值
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        # 返回扰动后输出的最大值和预测
        conf, pred = nnOutput.max(dim=1)

        # return pred, conf, msp
        return pred, conf, conf