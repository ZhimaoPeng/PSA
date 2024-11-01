
import time

import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from scood.losses import rew_ce, rew_sce,Extension,SupConLoss
from scood.utils import sort_array

from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer

import logging
from collections import Counter


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
        self.unlabeled_train_loader = unlabeled_train_loader
        self.labeled_aug_loader = labeled_aug_loader
        self.unlabeled_aug_loader = unlabeled_aug_loader
        self.unlabeled_aug_loader_mask = unlabeled_aug_loader_mask

        
        self.id_threshold = torch.tensor(id_threshold)
        
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
        
        self.num_clusters = num_clusters
        self.t = t
        
        self.lambda_oe = lambda_oe
        
        self.lambda_rep = lambda_rep
        self.hc = 1
        self.K = 128
        self.outs = [self.K] * self.hc
        self.pseudo_label_gt = self.unlabeled_train_loader.dataset.pseudo_label_gt
        self.id_quantile = id_quantile
        self.ood_quantile = ood_quantile
        self.exp_id = exp_id
        

    
    def train_epoch(self, epoch, output_dir):
        logging.basicConfig(filename=str(output_dir)+'/log.txt', level=logging.INFO)
        
        if self.use_threshold_training:
            if epoch == self.warm_up_epoch:
                if self.output_score_type == "energy_score":
                    self.id_threshold, self.ood_threshold= self.threshold_generation(epoch,self.id_quantile,self.ood_quantile,self.exp_id)
                logging.info(f"id_threshold: {self.id_threshold} ",)
                logging.info(f"ood_threshold: {self.ood_threshold} ",)
            
            if self.pseudo_generation_type == "per_epoch":
                if epoch >= self.warm_up_epoch:
                    per_class_weights = self.pseudo_generation(epoch, self.id_threshold, self.ood_threshold)
            else:
                per_class_weights = None
            if epoch >= self.warm_up_epoch:
                metrics = self._compute_loss(epoch,loss_fun_type = self.loss_fun_type,per_class_weights = per_class_weights)
            else:
                metrics = self._compute_loss(epoch)
            
            if epoch == (self.epochs-1):
                self.pseudo_generation(epoch, self.id_threshold, self.ood_threshold,save_ood_mask=True,output_dir = output_dir)
        if self.use_balanced_fine_tuning:
            metrics = self._compute_loss_balanced(epoch,loss_fun_type = self.loss_fun_type)

        return metrics



    def _compute_loss(self, epoch, loss_fun_type = "full", per_class_weights = None):
        self.net.train()  # enter train mode
        loss_avg = 0.0
        train_dataiter = iter(self.labeled_aug_loader)
        unlabeled_dataiter = iter(self.unlabeled_aug_loader)

        criterion_rep = Extension(temperature=self.t, scale_by_temperature=False)
        contrastive_loss = SupConLoss()

        # logging.info(f"Should_use_Pseu_label {np.unique(self.unlabeled_train_loader.dataset.pseudo_label,return_counts=True)}",)
        # logging.info(f"Actual Pseu_label {np.unique(self.unlabeled_aug_loader.dataset.pseudo_label,return_counts=True)}",)

        for train_step in range(1, len(train_dataiter) + 1):  
            batch = next(train_dataiter) 
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_aug_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            labeled_aug_data = batch["aug_data"]
            q_labeled_aug_data = labeled_aug_data[0].cuda()
            k_labeled_aug_data = labeled_aug_data[1].cuda()

            unlabeled_aug_data = unlabeled_batch["aug_data"]
            sclabel = unlabeled_batch["sc_label"].cuda()
            pseudo_label_pred = unlabeled_batch["pseudo_label_pred"].cuda()

            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()

            # sclabel = unlabeled_batch["sc_label"]

            if "_cl" in self.contra_loss_type:
                data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data, k_labeled_aug_data, k_unlabeled_aug_data])
            else:
                data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data])

            N1 = len(q_labeled_aug_data)
            num_ulb = len(q_unlabeled_aug_data)

            con_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)  
            
            con_logits_rep_k = con_logits_rep[N1+num_ulb:2*(N1+num_ulb)]
            con_logits_rep = con_logits_rep[:N1+num_ulb]
            con_logits_cls = con_logits_cls[:N1+num_ulb]

            if self.pseudo_generation_type == "per_iter":
                if epoch >= self.warm_up_epoch:
                    if self.use_threshold_training:
                        with torch.no_grad():
                            unlab_logits_cls = con_logits_cls[N1:N1+num_ulb]
                            unlab_energy_score = torch.logsumexp(unlab_logits_cls.detach(), dim=1)
                            max_logits, max_index = torch.max(unlab_logits_cls.detach(), dim=-1)
                            # unlab_id_threshold = self.id_threshold.cuda()[max_index]
                            # unlab_ood_threshold = self.ood_threshold.cuda()[max_index]
                            unlab_id_threshold = self.id_threshold.cuda()
                            unlab_ood_threshold = self.ood_threshold.cuda()
                            id_mask = (unlab_energy_score >= unlab_id_threshold)
                            ood_mask = (unlab_energy_score <= unlab_ood_threshold)

                            all_unlab_mask = torch.logical_or(id_mask, ood_mask)
                            lab_mask = torch.ones(len(batch["label"]), dtype=torch.bool).cuda()
                            training_sample_mask = torch.cat([lab_mask, all_unlab_mask]) 
                            indices = torch.where(id_mask == True)[0]
                            if len(indices) != 0:
                                unlabeled_batch["pseudo_label"].cuda()[indices] = max_index[indices]

            
            if "_cl" in self.contra_loss_type:
                # con_logits_rep_k = con_logits_rep[N1+num_ulb:2*(N1+num_ulb)]
                f1 = F.normalize(con_logits_rep, dim=1)
                f2 = F.normalize(con_logits_rep_k, dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
           

            concat_label = torch.cat([batch["label"], unlabeled_batch["pseudo_label"].type_as(
                batch["label"]), ]).cuda()
            

            if self.contra_loss_type == 'all_scl':
                if self.use_threshold_training:
                    if epoch < self.warm_up_epoch:
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


            # loss_rep = criterion_rep(con_logits_rep, concat_label.cuda())
            logits_augcls, logits_oe_augcls = con_logits_cls[:N1], con_logits_cls[N1:]
            
            cluster_ID_label = unlabeled_batch["pseudo_label"]
            cluster_ID_label = cluster_ID_label.type_as(batch["label"])

            '''standard CE loss(labeled ID+cluster ID)'''
            loss_cls = F.cross_entropy(con_logits_cls[concat_label > -1] / 0.5, concat_label[
                concat_label > -1].cuda(), ) + 0.3 * F.cross_entropy(logits_augcls / 0.5, batch["label"].cuda(), ) 
            # oe loss
            concat_softlabel = torch.cat([batch["soft_label"], unlabeled_batch["pseudo_softlabel"]]).cuda()

            if self.use_threshold_training:
                if epoch < self.warm_up_epoch:
                    loss_oe = 0
                else:
                    if torch.sum(concat_label == -1) == 0:
                        loss_oe = 0
                    else:
                        if self.pseudo_generation_type == "per_iter":
                            loss_oe = rew_sce(con_logits_cls[N1:][ood_mask], concat_softlabel[N1:][ood_mask].cuda(), )
                        else:
                            if self.use_oe_type == "balanced_oe":
                                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), per_class_weights,pseudo_label_pred[unlabeled_batch["pseudo_label"]== -1])
                            else:
                                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )

            else:
                loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )

            
            
            if epoch < self.warm_up_epoch:
                loss = loss_cls + self.lambda_rep * loss_rep
                # print('aaa')
            else:
                loss = loss_cls + self.lambda_rep * loss_rep + 0.5 * loss_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step()  
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            
        
        
        metrics = {}
        metrics["train_loss"] = loss_avg
        return metrics
    
    
    def _compute_loss_balanced(self, epoch,loss_fun_type = "full", per_class_weights = None):

        self.net.train()  # enter train mode
        loss_avg = 0.0
        train_dataiter = iter(self.labeled_aug_loader)
        unlabeled_dataiter = iter(self.unlabeled_aug_loader)

        criterion_rep = Extension(temperature=self.t, scale_by_temperature=False)
        contrastive_loss = SupConLoss()
        
        for train_step in range(1, len(train_dataiter) + 1):  
            batch = next(train_dataiter) 
            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.unlabeled_aug_loader)
                unlabeled_batch = next(unlabeled_dataiter)
            
            labeled_aug_data = batch["aug_data"]
            q_labeled_aug_data = labeled_aug_data[0].cuda()
            k_labeled_aug_data = labeled_aug_data[1].cuda()

            
            unlabeled_aug_data = unlabeled_batch["aug_data"]

            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()

            data = torch.cat([q_labeled_aug_data, q_unlabeled_aug_data])

            N1 = len(q_labeled_aug_data)
            num_ulb = len(q_unlabeled_aug_data)

            con_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True) 

            con_logits_rep = con_logits_rep[:N1+num_ulb]
            con_logits_cls = con_logits_cls[:N1+num_ulb]

            concat_label = torch.cat([batch["label"], unlabeled_batch["pseudo_label"].type_as(
                batch["label"]), ]).cuda()

            logits_augcls, logits_oe_augcls = con_logits_cls[:N1], con_logits_cls[N1:]

            '''standard CE loss(labeled ID+cluster ID)'''
            loss_cls = F.cross_entropy(con_logits_cls[concat_label > -1] / 0.5, concat_label[
                concat_label > -1].cuda(), ) + 0.3 * F.cross_entropy(logits_augcls / 0.5, batch["label"].cuda(), )

            # oe loss
            concat_softlabel = torch.cat([batch["soft_label"], unlabeled_batch["pseudo_softlabel"]]).cuda()
            loss_oe = rew_sce(con_logits_cls[concat_label == -1], concat_softlabel[concat_label == -1].cuda(), )
            loss_rep = criterion_rep(con_logits_rep[concat_label != -2], concat_label[concat_label != -2].cuda())
            
            
            loss = loss_cls + self.lambda_rep * loss_rep + 0.5 * loss_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step()  
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        # if (epoch==3):
            # print('aaa')
        metrics = {}
        metrics["train_loss"] = loss_avg
        return metrics
    
    def threshold_generation(self, epoch, id_quantile, ood_quantile, exp_id):
        
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
            self.net.eval()  # enter train mode
            unlabeled_dataiter = iter(self.labeled_aug_loader)
            for train_step in range(1, len(unlabeled_dataiter) + 1):
                unlabeled_batch = next(unlabeled_dataiter)
                unlabeled_aug_data = unlabeled_batch["aug_data"]
                q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
                k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()
                data = q_unlabeled_aug_data
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
    
    def pseudo_generation(self, epoch, unlab_id_threshold, unlab_ood_threshold,save_ood_mask = False,output_dir = None):
        all_unlab_energy_score = []
        all_sc_labels = []
        all_pred_labels = []
        per_class_image_num = []
        per_class_weights = []
        self.net.eval()  # enter train mode
        unlabeled_dataiter = iter(self.unlabeled_aug_loader_mask)
        all_unlab_num = 0
        

        for train_step in range(1, len(unlabeled_dataiter) + 1): 
            unlabeled_batch = next(unlabeled_dataiter)
            unlabeled_aug_data = unlabeled_batch["aug_data"]
            sclabel = unlabeled_batch["sc_label"].cuda()
            q_unlabeled_aug_data = unlabeled_aug_data[0].cuda()
            k_unlabeled_aug_data = unlabeled_aug_data[1].cuda()
            data = q_unlabeled_aug_data
            unlab_logits_cls, _, _, con_logits_rep = self.net(data, return_feature=True, return_aux=True)
            if self.output_score_type == "energy_score":
                unlab_energy_score = torch.logsumexp(unlab_logits_cls.detach(), dim=1)
            else:
                unlab_energy_score,_ = torch.max(torch.softmax(unlab_logits_cls.detach(), dim=1), dim=-1)
                # print('aaa')
            max_logits, max_index = torch.max(unlab_logits_cls.detach(), dim=-1)
            
            all_unlab_energy_score.append(unlab_energy_score)
            all_sc_labels.append(sclabel)
            all_pred_labels.append(max_index)
            all_unlab_num += len(data)
        
        

        all_unlab_energy_score = torch.concatenate(all_unlab_energy_score)
        all_sc_labels = torch.concatenate(all_sc_labels)
        all_pred_labels =  torch.concatenate(all_pred_labels)

        if self.output_score_type == "sort":
            unlab_id_threshold = torch.quantile(all_unlab_energy_score, unlab_id_threshold.cuda())
            unlab_ood_threshold = torch.quantile(all_unlab_energy_score, unlab_ood_threshold.cuda())
            
        
        id_mask = all_unlab_energy_score >= unlab_id_threshold
        ood_mask = all_unlab_energy_score <= unlab_ood_threshold
        discard_mask = torch.logical_not(torch.logical_or(id_mask,ood_mask)) 
        
        if save_ood_mask == True:
            np.save(output_dir/f"final_ood_mask.npy",ood_mask.cpu().numpy())
            np.save(output_dir/f"final_id_mask.npy",id_mask.cpu().numpy())
            np.save(output_dir/f"id_pseudo_label.npy",all_pred_labels[id_mask].cpu().numpy())
            np.save(output_dir/f"all_unlab_energy_score.npy",all_unlab_energy_score.cpu().numpy())

        
        # ood_pseudo_label = all_pred_labels[ood_mask]
        ood_pseudo_label = self.unlabeled_aug_loader.dataset.pseudo_label_pred[ood_mask.cpu().numpy()]

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

        
        if self.use_id_type == "use_id":
            self.unlabeled_aug_loader.dataset.pseudo_label[id_mask.cpu().numpy()] = all_pred_labels[id_mask].cpu().numpy()
            self.unlabeled_aug_loader.dataset.pseudo_label[discard_mask.cpu().numpy()] = -2
        elif self.use_id_type == "use_discard_id":
             self.unlabeled_aug_loader.dataset.pseudo_label[non_ood_mask.cpu().numpy()] = all_pred_labels[non_ood_mask].cpu().numpy()
        elif self.use_id_type == "discard_id":
            self.unlabeled_aug_loader.dataset.pseudo_label[non_ood_mask.cpu().numpy()] = -2
        
        return per_class_weights




