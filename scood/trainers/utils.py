from typing import Any, Dict
from torch import float64

import torch.nn as nn
from torch.utils.data import DataLoader

from .ETtrainer import ETtrainer


def get_ETtrainer(
    net: nn.Module,
    labeled_train_loader: DataLoader,
    unlabeled_train_loader: DataLoader,
    labeled_aug_loader: DataLoader,
    unlabeled_aug_loader: DataLoader,
    unlabeled_aug_loader_mask: DataLoader,
    use_threshold_training,
    use_balanced_fine_tuning,
    lamda: float, 
    contra_loss_type: str,
    optim_args: Dict[str, Any],
    trainer_args: Dict[str, Any],
):  
    return ETtrainer(
        net, labeled_train_loader, unlabeled_train_loader, labeled_aug_loader, unlabeled_aug_loader, unlabeled_aug_loader_mask,use_threshold_training,use_balanced_fine_tuning,lamda,contra_loss_type, **optim_args, **trainer_args
    )
