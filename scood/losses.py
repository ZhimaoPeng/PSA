import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Extension(nn.Module):

    def __init__(self, temperature=0., scale_by_temperature=True):
        super(Extension, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # 对特征进行归一化
        features = F.normalize(features, p=2, dim=1)
        # 获取batch中标注和无标注样本的总数量
        batch_size = features.shape[0]
        if labels is not None and mask is not None: 
            raise ValueError('Cannot define both `labels` and `mask`')
        # 如果标签不存在并且遮罩也不存在,退化为自监督对比学习
        elif labels is None and mask is None:  
            # 获取自监督对比学习的遮罩
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        # 如果标签存在
        elif labels is not None:  
            # 转换标签的尺寸
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # 获取监督对比损失的遮罩
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # create mask
        logits_mask = torch.ones_like(mask) - (torch.eye(batch_size)).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
       
        num_positives_per_row = torch.sum(positives_mask, axis=1)  
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.05, contrast_mode='all',
                 base_temperature=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast[anchor_dot_contrast==float('inf')] = 1

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits[logits==float('inf')] = 1


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask[mask==float('inf')] = 1

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob[log_prob==float('inf')] = 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# 计算软交叉熵损失
class SoftCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, label, weight=None):
        assert logit.size() == label.size(), "logit.size() != label.size()" 
        dim = logit.dim() 
        max_logit = logit.max(dim - 1, keepdim=True)[0] 
        logit = logit - max_logit 
                                  
        exp_logit = logit.exp() 
        exp_sum = exp_logit.sum(dim - 1, keepdim=True)
        prob = exp_logit / exp_sum  
        log_exp_sum = exp_sum.log() 
        neg_log_prob = log_exp_sum - logit 

        if weight is None:
            weighted_label = label
        else:
            if weight.size() != logit.size():
                raise ValueError(
                    "since logit.size() = {}, weight.size() should be ({},), but got {}".format(
                        logit.size(),
                        logit.size(),
                        weight.size(),
                    )
                )
            size = [1] * label.dim()
            size[-1] = label.size(-1)
            # 获取每类的权重加权后的标签
            # weighted_label = label * weight.view(size)
            weighted_label = label * weight
        ctx.save_for_backward(weighted_label, prob)
        out = (neg_log_prob * weighted_label).sum(dim - 1)  
                                                            
        return out

    @staticmethod
    def backward(ctx, grad_output): 
        weighted_label, prob = ctx.saved_tensors
        old_size = weighted_label.size() 
        # num_classes
        K = old_size[-1]  
        # batch_size
        B = weighted_label.numel() // K 
                                         

        grad_output = grad_output.view(B, 1) 
        weighted_label = weighted_label.view(B, K)
        prob = prob.view(B, K)
        grad_input = grad_output * (prob * weighted_label.sum(1, True) - weighted_label)
        grad_input = grad_input.view(old_size) 
        return grad_input, None, None


# 计算软交叉熵损失
def soft_cross_entropy(logit, label, weight=None, reduce=None, reduction="mean"):
    if weight is not None and weight.requires_grad:
        raise RuntimeError("gradient for weight is not supported")
    losses = SoftCrossEntropyFunction.apply(logit, label, weight) 
    reduction = {
        True: "mean",
        False: "none",
        None: reduction,
    }[reduce]  
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses 
    else:
        raise ValueError("invalid value for reduction: {}".format(reduction))


# 计算软交叉熵损失
class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, reduce=None, reduction="mean"):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, logit, label, weight=None):
        if weight is None:
            weight = self.weight
        return soft_cross_entropy(logit, label, weight, self.reduce, self.reduction)


def rew_ce(logits, labels, sample_weights):
    losses = F.cross_entropy(logits, labels, reduction="none") 
    return (losses * sample_weights.type_as(losses)).mean() 


# 计算软交叉熵损失
def rew_sce(logits, soft_labels,per_class_weights=None,pseudo_label_pred = None):
    if pseudo_label_pred is not None:
        per_class_weights = torch.tensor(per_class_weights).cuda()

        sample_weights = per_class_weights[pseudo_label_pred]
        sample_weights = sample_weights.unsqueeze(1).repeat(1,logits.size(1))
        losses = soft_cross_entropy(logits, soft_labels, sample_weights,reduce=False)
    else:
        # 计算软交叉熵损失
        losses = soft_cross_entropy(logits, soft_labels, reduce=False)
    #return torch.mean(losses * sample_weights.type_as(losses)) 
    # 返回损失均值
    return torch.mean(losses.type_as(losses))

def prob_2_entropy(softmax_prob):
    n, c = softmax_prob.size()
    entropy = -torch.mul(softmax_prob, torch.log2(softmax_prob + 1e-30))
    return torch.mean(entropy, dim=1) / np.log2(c)


def log_sum_exp(value, dim=None, keepdim=False):   #value is logits
    weight_energy = torch.nn.Linear(10, 1).cuda()
    torch.nn.init.uniform_(weight_energy.weight)
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)  #MAX_logits
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)   #压缩维度
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
