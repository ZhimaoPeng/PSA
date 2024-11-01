import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def cosine_annealing(step, total_steps, warmup_steps, lr_max, lr_min): 
    if step < warmup_steps:
        return (step / warmup_steps)
    else:
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
   

class BaseTrainer:
    def __init__(
        self,
        net: nn.Module,
        labeled_train_loader: DataLoader,
        learning_rate: float = 0.1,
        min_lr: float = 1e-6,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        epochs: int = 100,
        warmup_epochs: int = 30,
    ) -> None:
        self.net = net
        self.labeled_train_loader = labeled_train_loader

        # 优化器为SGD
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        # 获取余弦退火学习率
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(labeled_train_loader),
                warmup_epochs * len(labeled_train_loader),
                1,  # since lr_lambda computes multiplicative factor
                min_lr / learning_rate,
            ),
        )


    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.labeled_train_loader)

        for train_step in range(1, len(train_dataiter) + 1):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            target = batch["label"].cuda()
            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics["train_loss"] = loss_avg

        return metrics

