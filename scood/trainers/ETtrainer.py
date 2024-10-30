# 时间相关
import time
# numpy相关
import numpy as np
# torch相关
import torch
# 神经网络相关
import torch.nn as nn
import torch.nn.functional as F
from scood.losses import rew_ce, rew_sce,Extension,SupConLoss
from scood.utils import sort_array
# 获取数据加载器
from torch.utils.data import DataLoader
# 获取基本训练器
from .base_trainer import BaseTrainer
# import subprocess
import logging
from collections import Counter

# 能量最优传输训练
class ETtrainer(BaseTrainer):
    def __init__(
            self,
            net: nn.Module,
            labeled_train_loader: DataLoader,
            unlabeled_train_loader: DataLoader,
            labeled_aug_loader: DataLoader,
            unlabeled_aug_loader: DataLoader,
            unlabeled_aug_loader_mask: DataLoader,
            use_threshold_training,
            use_balanced_fine_tuning,
            lamda: float,
            contra_loss_type: str = "all_scl",
            learning_rate: float = 0.1,
            min_lr: float = 1e-6,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            warm_up_epoch : int = 0,
            epochs: int = 100,
            num_clusters: int = 1000,
            t: float = 0.5,
            lambda_oe: float = 0.5,
            lambda_rep: float = 0.3,
            id_quantile: float = 0.9,
            ood_quantile: float = 0.2,
            exp_id: int = 0,
            use_id_type: str = "use_id",
            use_oe_type: str = "oe",
            beta: float = 0.999,
            use_fixed_threshold: bool = False,
            pseudo_generation_type: str = "per_epoch",
            output_score_type: str = "energy_score",
            id_threshold: float = 0.9,
            ood_threshold: float = 0.2,
            loss_fun_type: str = "full",

    ) -> None:
        super().__init__(
            net,
            labeled_train_loader,
            learning_rate=learning_rate,
            min_lr=min_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
        )

        self.epochs = epochs
        # 获取无标注样本的加载器
        self.unlabeled_train_loader = unlabeled_train_loader
        self.labeled_aug_loader = labeled_aug_loader
        self.unlabeled_aug_loader = unlabeled_aug_loader
        self.unlabeled_aug_loader_mask = unlabeled_aug_loader_mask

        # 获取ID阈值
        self.id_threshold = torch.tensor(id_threshold)
        # 获取OOD阈值
        self.ood_threshold = torch.tensor(ood_threshold)
        self.loss_fun_type = loss_fun_type

        self.use_threshold_training = use_threshold_training
        self.use_balanced_fine_tuning = use_balanced_fine_tuning
        self.lamda = lamda

        self.contra_loss_type = contra_loss_type
        self.use_id_type = use_id_type
        self.use_oe_type = use_oe_type
        self.beta = beta
        self.use_fixed_threshold = use_fixed_threshold
        self.pseudo_generation_type = pseudo_generation_type
        self.output_score_type = output_score_type

        self.warm_up_epoch = warm_up_epoch
        # 获取聚类簇的数量
        self.num_clusters = num_clusters
        self.t = t
        # 获取样本暴露损失的权重
        self.lambda_oe = lambda_oe
        # 获取表示学习损失的权重
        self.lambda_rep = lambda_rep
        self.hc = 1
        self.K = 128
        # 获取输出的维度
        self.outs = [self.K] * self.hc
        # 获取OOD样本的GT伪标签
        self.pseudo_label_gt = self.unlabeled_train_loader.dataset.pseudo_label_gt
        self.id_quantile = id_quantile
        self.ood_quantile = ood_quantile
        self.exp_id = exp_id
        

    # 进行一个周期的训练
    def train_epoch(self, epoch, output_dir):
        # 获取日志的基本配置
        logging.basicConfig(filename=str(output_dir)+'/log.txt', level=logging.INFO)
        
        # 如果使用阈值训练
        if self.use_threshold_training:
            # 如果达到warmup的周期
            if epoch == self.warm_up_epoch:
                if self.output_score_type == "energy_score":
                    # 进行阈值的生成
                    self.id_threshold, self.ood_threshold= self.threshold_generation(epoch,self.id_quantile,self.ood_quantile,self.exp_id)
                logging.info(f"id_threshold: {self.id_threshold} ",)
                logging.info(f"ood_threshold: {self.ood_threshold} ",)
            
            # 如果每周期生成一次伪标签
            if self.pseudo_generation_type == "per_epoch":
                # 如果超过warmup的周期
                if epoch >= self.warm_up_epoch:
                    # 获取伪标签的遮罩
                    per_class_weights = self.pseudo_generation(epoch, self.id_threshold, self.ood_threshold)
            else:
                per_class_weights = None
            # 计算损失值
            if epoch >= self.warm_up_epoch:
                metrics = self._compute_loss(epoch,loss_fun_type = self.loss_fun_type,per_class_weights = per_class_weights)
            else:
                metrics = self._compute_loss(epoch)
            
            # 如果是最后一个周期
            if epoch == (self.epochs-1):
                # 保存伪ID和OOD样本的遮罩
                self.pseudo_generation(epoch, self.id_threshold, self.ood_threshold,save_ood_mask=True,output_dir = output_dir)
        # 如果进行平衡微调
        if self.use_balanced_fine_tuning:
            # 进行平衡微调
            metrics = self._compute_loss_balanced(epoch,loss_fun_type = self.loss_fun_type)

        return metrics



    # 计算损失值
    def _compute_loss(self, epoch, loss_fun_type = "full", per_class_weights = None):
        # 设置网络为训练模式
        self.net.train()  # enter train mode
        loss_avg = 0.0
        # 获取增广变换后的标注样本
        train_dataiter = iter(self.labeled_aug_loader)
        # 获取增广变换后的无标注样本
        unlabeled_dataiter = iter(self.unlabeled_aug_loader)

        

        # 获取表示相关的损失函数
        criterion_rep = Extension(temperature=self.t, scale_by_temperature=False)
        contrastive_loss = SupConLoss()

        # logging.info(f"Should_use_Pseu_label {np.unique(self.unlabeled_train_loader.dataset.pseudo_label,return_counts=True)}",)
        # logging.info(f"Actual Pseu_label {np.unique(self.unlabeled_aug_loader.dataset.pseudo_label,return_counts=True)}",)

        # 遍历所有的训练样本
        for train_step in range(1, len(train_dataiter) + 1):  
            # 获取增广变换后的标注样本
            batch = next(train_dataiter) 
            # 获取增广变换后的无标注样本
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_aug_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            # 获取标注样本的两个增广变换
            labeled_aug_data = batch["aug_data"]
            # 将标注增广样本送入到GPU中
            q_labeled_aug_data = labeled_aug_data[0].cuda()
            k_labeled_aug_data = labeled_aug_data[1].cuda()

            # 获取无标注图像的两个增广变换
            unlabeled_aug_data = unlabeled_batch["aug_data"]
            sclabel = unlabeled_batch["sc_label"].cuda()
            pseudo_label_pred = unlabeled_batch["pseudo_label_pred"].cuda()

            # 将无标注增广样本送入到GPU中
            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()

            # sclabel = unlabeled_batch["sc_label"]

            # 如果使用无监督对比损失
            if "_cl" in self.contra_loss_type:
                # 将两个增广的训练数据拼接起来
                data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data, k_labeled_aug_data, k_unlabeled_aug_data])
            else:
                # 将标注样本和无标注样本拼接起来
                data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data])

            # 获取标注样本的数量
            N1 = len(q_labeled_aug_data)
            # 获取无标注样本的数量
            num_ulb = len(q_unlabeled_aug_data)

            # 将数据送入到网络中，得到所有样本的logits和特征表示
            con_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)  
            
        
            # 获取标注和无标注样本增广变换的输出
            con_logits_rep_k = con_logits_rep[N1+num_ulb:2*(N1+num_ulb)]
            con_logits_rep = con_logits_rep[:N1+num_ulb]
            con_logits_cls = con_logits_cls[:N1+num_ulb]

            # 生成伪标签的类型，是每次迭代还是每个周期
            if self.pseudo_generation_type == "per_iter":
                if epoch >= self.warm_up_epoch:
                    if self.use_threshold_training:
                        with torch.no_grad():
                            unlab_logits_cls = con_logits_cls[N1:N1+num_ulb]
                            unlab_energy_score = torch.logsumexp(unlab_logits_cls.detach(), dim=1)
                            # 获取基于预测的最大概率和对应的类别号
                            max_logits, max_index = torch.max(unlab_logits_cls.detach(), dim=-1)
                            # unlab_id_threshold = self.id_threshold.cuda()[max_index]
                            # unlab_ood_threshold = self.ood_threshold.cuda()[max_index]
                            unlab_id_threshold = self.id_threshold.cuda()
                            unlab_ood_threshold = self.ood_threshold.cuda()
                            # 获取ID样本的遮罩
                            id_mask = (unlab_energy_score >= unlab_id_threshold)
                            # 获取OOD样本的遮罩
                            ood_mask = (unlab_energy_score <= unlab_ood_threshold)

                            # 获取使用的无标注样本的遮罩
                            all_unlab_mask = torch.logical_or(id_mask, ood_mask)
                            lab_mask = torch.ones(len(batch["label"]), dtype=torch.bool).cuda()
                            # 获取所有参与训练样本的遮罩
                            training_sample_mask = torch.cat([lab_mask, all_unlab_mask]) 
                            indices = torch.where(id_mask == True)[0]
                            if len(indices) != 0:
                                unlabeled_batch["pseudo_label"].cuda()[indices] = max_index[indices]

            # 如果使用无监督对比学习
            if "_cl" in self.contra_loss_type:
                # con_logits_rep_k = con_logits_rep[N1+num_ulb:2*(N1+num_ulb)]
                # 获取两个视图的特征
                f1 = F.normalize(con_logits_rep, dim=1)
                f2 = F.normalize(con_logits_rep_k, dim=1)
                # 将所有样本两个视图的特征拼接起来
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
           
            # 将标注数据的标签和无标注数据的伪标签拼接起来，无标注数据为伪标签均为-1
            concat_label = torch.cat([batch["label"], unlabeled_batch["pseudo_label"].type_as(
                batch["label"]), ]).cuda()
            
            # 计算所有样本的监督对比损失
            if self.contra_loss_type == 'all_scl':
                # 如果使用阈值进行训练
                if self.use_threshold_training:
                    # 如果进行warmup
                    if epoch < self.warm_up_epoch:
                        # warmup期间只进行ID样本的监督训练
                        loss_rep = criterion_rep(con_logits_rep[:N1], concat_label[:N1].cuda())
                    else:
                        if self.pseudo_generation_type == "per_iter":
                            loss_rep = criterion_rep(con_logits_rep[training_sample_mask], concat_label[training_sample_mask].cuda())
                        else:
                            loss_rep = criterion_rep(con_logits_rep[concat_label != -2], concat_label[concat_label != -2].cuda())
                else:
                    loss_rep = criterion_rep(con_logits_rep, concat_label.cuda())
            elif self.contra_loss_type == 'none':
                loss_rep = 0


            # 计算所有样本的监督对比损失
            # loss_rep = criterion_rep(con_logits_rep, concat_label.cuda())
            # 获取标注样本的输出logits
            logits_augcls, logits_oe_augcls = con_logits_cls[:N1], con_logits_cls[N1:]
            
            cluster_ID_label = unlabeled_batch["pseudo_label"]
            cluster_ID_label = cluster_ID_label.type_as(batch["label"])

            # 计算聚类得到的ID样本和真实ID样本的联合交叉熵损失
            '''standard CE loss(labeled ID+cluster ID)'''
            loss_cls = F.cross_entropy(con_logits_cls[concat_label > -1] / 0.5, concat_label[
                concat_label > -1].cuda(), ) + 0.3 * F.cross_entropy(logits_augcls / 0.5, batch["label"].cuda(), ) 
            # oe loss
            # 获取所有样本的软标签
            concat_softlabel = torch.cat([batch["soft_label"], unlabeled_batch["pseudo_softlabel"]]).cuda()

            if self.use_threshold_training:
                if epoch < self.warm_up_epoch:
                    loss_oe = 0
                else:
                    if torch.sum(concat_label == -1) == 0:
                        loss_oe = 0
                    else:
                        # 如果每次迭代生成伪标签
                        if self.pseudo_generation_type == "per_iter":
                            # 为无标注样本计算样本暴露损失
                            loss_oe = rew_sce(con_logits_cls[N1:][ood_mask], concat_softlabel[N1:][ood_mask].cuda(), )
                        # 如果每个周期生成伪标签
                        else:
                            if self.use_oe_type == "balanced_oe":
                                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), per_class_weights,pseudo_label_pred[unlabeled_batch["pseudo_label"]== -1])
                            else:
                                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )

            else:
                # 计算OOD暴露损失，该损失使OOD样本的熵最大化
                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )

            
            
            if epoch < self.warm_up_epoch:
                loss = loss_cls + self.lambda_rep * loss_rep
                # print('aaa')
            else:
                # 获取分类损失，表示损失和OOD暴露损失
                loss = loss_cls + self.lambda_rep * loss_rep + 0.5 * loss_oe

            # backward
            # 梯度置零
            self.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降
            self.optimizer.step() 
            self.scheduler.step()  
            # 不track梯度
            with torch.no_grad():
                # 计算平均损失
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            
        
        
        metrics = {}
        metrics["train_loss"] = loss_avg
        return metrics
    
    # 计算损失值
    def _compute_loss_balanced(self, epoch,loss_fun_type = "full", per_class_weights = None):

        # 设置网络为训练模式
        self.net.train()  # enter train mode
        loss_avg = 0.0
        # 获取增广变换后的标注样本
        train_dataiter = iter(self.labeled_aug_loader)
        # 获取增广变换后的无标注样本
        unlabeled_dataiter = iter(self.unlabeled_aug_loader)

        # 获取表示相关的损失函数
        criterion_rep = Extension(temperature=self.t, scale_by_temperature=False)
        contrastive_loss = SupConLoss()
        
        # 遍历所有的训练样本
        for train_step in range(1, len(train_dataiter) + 1):  
            # 获取增广变换后的标注样本
            batch = next(train_dataiter) 
            # 获取增广变换后的无标注样本
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_aug_loader)
                unlabeled_batch = next(unlabeled_dataiter)
            
            # 获取标注样本的两个增广变换
            labeled_aug_data = batch["aug_data"]
            # 将标注增广样本送入到GPU中
            q_labeled_aug_data = labeled_aug_data[0].cuda()
            k_labeled_aug_data = labeled_aug_data[1].cuda()

            # 获取无标注图像的两个增广变换
            unlabeled_aug_data = unlabeled_batch["aug_data"]

            # 将无标注增广样本送入到GPU中
            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()

            data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data])

            # 获取标注样本的数量
            N1 = len(q_labeled_aug_data)
            # 获取无标注样本的数量
            num_ulb = len(q_unlabeled_aug_data)

            # 将数据送入到网络中，得到所有样本的logits和特征表示
            con_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True) 

            # 获取标注和无标注样本增广变换的输出
            con_logits_rep = con_logits_rep[:N1+num_ulb]
            con_logits_cls = con_logits_cls[:N1+num_ulb]

            # 将标注数据的标签和无标注数据的伪标签拼接起来，无标注数据为伪标签均为-1
            concat_label = torch.cat([batch["label"], unlabeled_batch["pseudo_label"].type_as(
                batch["label"]), ]).cuda()

            logits_augcls, logits_oe_augcls = con_logits_cls[:N1], con_logits_cls[N1:]

             # 计算聚类得到的ID样本和真实ID样本的联合交叉熵损失
            '''standard CE loss(labeled ID+cluster ID)'''
            loss_cls = F.cross_entropy(con_logits_cls[concat_label > -1] / 0.5, concat_label[
                concat_label > -1].cuda(), ) + 0.3 * F.cross_entropy(logits_augcls / 0.5, batch["label"].cuda(), )

            # oe loss
            # 获取所有样本的软标签
            concat_softlabel = torch.cat([batch["soft_label"], unlabeled_batch["pseudo_softlabel"]]).cuda()
            loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )
            loss_rep = criterion_rep(con_logits_rep[concat_label != -2], concat_label[concat_label != -2].cuda())
            
            
            loss = loss_cls + self.lambda_rep * loss_rep + 0.5 * loss_oe

            # backward
            # 梯度置零
            self.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降
            self.optimizer.step() 
            self.scheduler.step()  
            # 不track梯度
            with torch.no_grad():
                # 计算平均损失
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        # if (epoch==3):
            # print('aaa')
        metrics = {}
        metrics["train_loss"] = loss_avg
        return metrics
    # 生成阈值
    def threshold_generation(self, epoch, id_quantile, ood_quantile, exp_id):
        
        # 如果使用固定的阈值
        if self.use_fixed_threshold == True:
            num_classes = self.unlabeled_aug_loader.dataset.num_classes
            if num_classes == 100:
                unlab_id_threshold = torch.tensor(8.165079975128172)
                unlab_ood_threshold = torch.tensor(4.9559777736663815)
            if num_classes == 10:
                unlab_id_threshold = torch.tensor(6.08179874420166)
                unlab_ood_threshold = torch.tensor(2.9806350469589233)  
        else:
            all_unlab_energy_score = []
            # 设置网络为训练模式
            self.net.eval()  # enter train mode
            # 获取增广变换后的无标注样本
            unlabeled_dataiter = iter(self.labeled_aug_loader)
            # 遍历所有的训练样本
            for train_step in range(1, len(unlabeled_dataiter) + 1):
                unlabeled_batch = next(unlabeled_dataiter)
                # 获取无标注图像的两个增广变换
                unlabeled_aug_data = unlabeled_batch["aug_data"]
                # 将无标注增广样本送入到GPU中
                q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
                k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()
                data = q_unlabeled_aug_data
                # 将数据送入到网络中，得到所有样本的logits和特征表示
                unlab_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)
                unlab_energy_score = torch.logsumexp(unlab_logits_cls.detach(), dim=1)
                all_unlab_energy_score.append(unlab_energy_score.cpu().numpy())
            all_unlab_energy_score = np.concatenate(all_unlab_energy_score)
            # np.save('saving_energy/all_lab_energy_score_epoch_{}_{}_{}_{}.npy'.format(epoch,id_quantile, ood_quantile, exp_id),all_unlab_energy_score)
            for i in np.arange(0,1.1,0.1):
                logging.info(f"{np.quantile(all_unlab_energy_score,i)} quantile {i}!",)
            unlab_id_threshold = torch.tensor(np.quantile(all_unlab_energy_score,self.id_quantile))
            unlab_ood_threshold = torch.tensor(np.quantile(all_unlab_energy_score,self.ood_quantile))
        return unlab_id_threshold,unlab_ood_threshold
    
    # 生成伪标签
    def pseudo_generation(self, epoch, unlab_id_threshold, unlab_ood_threshold,save_ood_mask = False,output_dir = None):
        all_unlab_energy_score = []
        all_sc_labels = []
        all_pred_labels = []
        per_class_image_num = []
        per_class_weights = []
        # 设置网络为训练模式
        self.net.eval()  # enter train mode
        # 获取增广变换后的无标注样本
        unlabeled_dataiter = iter(self.unlabeled_aug_loader_mask)
        all_unlab_num = 0
        
        # 遍历所有的训练样本
        for train_step in range(1, len(unlabeled_dataiter) + 1): 
            unlabeled_batch = next(unlabeled_dataiter)
            # 获取无标注图像的两个增广变换
            unlabeled_aug_data = unlabeled_batch["aug_data"]
            sclabel = unlabeled_batch["sc_label"].cuda()
            # 将无标注增广样本送入到GPU中
            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()
            data = q_unlabeled_aug_data
            # 将数据送入到网络中，得到所有样本的logits和特征表示
            unlab_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)
            if self.output_score_type == "energy_score":
                # 计算无标注样本的能量值
                unlab_energy_score = torch.logsumexp(unlab_logits_cls.detach(), dim=1)
            else:
                unlab_energy_score,_ = torch.max(torch.softmax(unlab_logits_cls.detach(), dim=1), dim=-1)
                # print('aaa')
            # 获取基于预测的最大概率和对应的类别号
            max_logits, max_index = torch.max(unlab_logits_cls.detach(), dim=-1)
            
            # 将无标注样本的能量值保存起来
            all_unlab_energy_score.append(unlab_energy_score)
            # 将语义一致样本标签保存起来
            all_sc_labels.append(sclabel)
            # 将无标注样本的预测标签保存起来
            all_pred_labels.append(max_index)

            # 获取所有无标注样本的数量
            all_unlab_num += len(data)
        
        

        # 将所有无标注样本的能量分数拼接起来
        all_unlab_energy_score = torch.concatenate(all_unlab_energy_score)
        # 将所有语义一致样本拼接起来
        all_sc_labels = torch.concatenate(all_sc_labels)
        # 将无标注样本的预测标签拼接起来
        all_pred_labels =  torch.concatenate(all_pred_labels)

        if self.output_score_type == "sort":
            unlab_id_threshold = torch.quantile(all_unlab_energy_score, unlab_id_threshold.cuda())
            unlab_ood_threshold = torch.quantile(all_unlab_energy_score, unlab_ood_threshold.cuda())
            
        # 获取ID样本的遮罩
        id_mask = all_unlab_energy_score >= unlab_id_threshold
        # 获取OOD样本的遮罩
        ood_mask = all_unlab_energy_score <= unlab_ood_threshold
        # 获取丢弃样本的遮罩
        discard_mask = torch.logical_not(torch.logical_or(id_mask,ood_mask)) 
        
        if save_ood_mask == True:
            np.save(output_dir/f"final_ood_mask.npy",ood_mask.cpu().numpy())
            np.save(output_dir/f"final_id_mask.npy",id_mask.cpu().numpy())
            np.save(output_dir/f"id_pseudo_label.npy",all_pred_labels[id_mask].cpu().numpy())
            np.save(output_dir/f"all_unlab_energy_score.npy",all_unlab_energy_score.cpu().numpy())

        
        # ood_pseudo_label = all_pred_labels[ood_mask]
        ood_pseudo_label = self.unlabeled_aug_loader.dataset.pseudo_label_pred[ood_mask.cpu().numpy()]

        # 获取非OOD样本的遮罩
        non_ood_mask = torch.logical_not(ood_mask)
        
        
        
        counter = Counter(ood_pseudo_label)
        sorted_samples_stats = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"{sorted_samples_stats} ood pseudo label stats!", )

        num_classes = self.unlabeled_aug_loader.dataset.num_classes
        for i in range(num_classes):
            per_class_image_num.append(len(np.where(ood_pseudo_label == i)[0]))
        
        effective_num = 1.0 - np.power(self.beta, per_class_image_num)
        for i in effective_num:
            if i == 0:
                weight = 0
            else:
                weight = (1.0 - self.beta) / i
            per_class_weights.append(weight)
        per_class_weights = np.array(per_class_weights)
        per_class_weights = per_class_weights / np.sum(per_class_weights) * num_classes
        
        
        logging.info(f"{per_class_image_num} per class image num!", )
        logging.info(f"{np.where(np.array(per_class_image_num)==0)[0]} missed classes!", )
        logging.info(f"{per_class_weights} per_class_weights!", )

        # 如果使用ID数据
        if self.use_id_type == "use_id":
            # 将ID数据的标签设置为预测的伪标签
            self.unlabeled_aug_loader.dataset.pseudo_label[id_mask.cpu().numpy()] = all_pred_labels[id_mask].cpu().numpy()
            # 将被丢弃的数据的标签设置为-2
            self.unlabeled_aug_loader.dataset.pseudo_label[discard_mask.cpu().numpy()] = -2
        elif self.use_id_type == "use_discard_id":
             self.unlabeled_aug_loader.dataset.pseudo_label[non_ood_mask.cpu().numpy()] = all_pred_labels[non_ood_mask].cpu().numpy()
        elif self.use_id_type == "discard_id":
            self.unlabeled_aug_loader.dataset.pseudo_label[non_ood_mask.cpu().numpy()] = -2
        
        return per_class_weights




