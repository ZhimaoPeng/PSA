"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F



def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    weights = []
    # 获取分母的值 每类样本数[2,3,1,2,2]
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    # 获取每类的权重值 array([0.500025  , 0.33336667, 1.        , 0.500025  , 0.500025  ])
    for i in effective_num:
        if i == 0:
            weight = 0
        else:
            weight = (1.0 - beta) / i
        weights.append(weight)
    weights = np.array(weights)
    # weights = (1.0 - beta) / np.array(effective_num)
    print('weights:',weights)
    # 对权重值进行加权
    weights = weights / np.sum(weights) * no_of_classes

    
    # 将标签转换为one-hot形式
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    # 将权重转换为张量
    weights = torch.tensor(weights).float()
    # 将权重扩充一个维度
    weights = weights.unsqueeze(0)
    # 获取每个样本对应的权重值
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    # 将样本权重扩充一个维度
    weights = weights.unsqueeze(1)
    # 将样本权重复制
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss



# 直接运行以下代码
if __name__ == '__main__':
    # 获取类别的数量
    no_of_classes = 5
    # 初始化样本的logits
    logits = torch.rand(10,no_of_classes).float()
    # 初始化标签
    labels = torch.randint(0,no_of_classes, size = (10,))
    # 初始化
    beta = 0.9
    # 初始化gamma值
    gamma = 2.0
    # 获取每类样本的数量
    samples_per_cls = [2,20,5,15,30]
    # 获取损失函数的类型
    loss_type = "focal"
    # 获取CB损失
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print(cb_loss)