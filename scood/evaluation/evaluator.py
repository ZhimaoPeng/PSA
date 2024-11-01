import csv
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from scood.postprocessors import BasePostprocessor
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
    ):
        self.net = net

    def inference(self, data_loader: DataLoader, postprocessor: BasePostprocessor):
        pred_list, conf_list, conf1_list, ddood_list, scood_list = [], [], [], [], []

        for batch in data_loader:
            data = batch["data"].cuda()   
            label = batch["label"].cuda()   
            sclabel = batch["sc_label"].cuda() 

            pred, conf, conf1 = postprocessor(self.net, data) 


            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                conf1_list.append(conf1[idx].cpu().tolist())
                ddood_list.append(label[idx].cpu().tolist())  
                scood_list.append(sclabel[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        conf1_list=np.array(conf1_list)
        ddood_list = np.array(ddood_list, dtype=int)
        scood_list = np.array(scood_list, dtype=int)

        return pred_list, conf_list, conf1_list, ddood_list, scood_list

    def eval_classification(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in data_loader: 
                data = batch["data"].cuda()
                target = batch["label"].cuda()

                # forward
                output = self.net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]  
                correct += pred.eq(target.data).sum().item() 

                # test loss average
                loss_avg += float(loss.data)

        metrics = {}
        metrics["test_loss"] = loss_avg / len(data_loader)  
        metrics["test_accuracy"] = correct / len(data_loader.dataset) 

        return metrics

    def eval_ood(
        self,
        id_data_loader: DataLoader,
        ood_data_loaders: List[DataLoader], 
        postprocessor: BasePostprocessor = None,
        method: str = "each",
        dataset_type: str = "scood",
        csv_path: str = None,
        output_dir: str = None,
    ):
        self.net.eval()

        if postprocessor is None:
            postprocessor = BasePostprocessor()
        
        # logging.basicConfig(filename=str(output_dir)+'/log.txt', level=logging.INFO)

        if method == "sel":
            results_matrix = []

            id_name = id_data_loader.dataset.name 

            logging.info(f"Performing inference on {id_name} dataset...")
            id_pred, id_conf, id_conf1, id_ddood, id_scood = self.inference(
                id_data_loader, postprocessor
            )

            for i, ood_dl in enumerate(ood_data_loaders):
                ood_name = ood_dl.dataset.name

                # logging.info(f"Performing inference on {ood_name} dataset...")
                ood_pred, ood_conf, ood_conf1, ood_ddood, ood_scood = self.inference(
                    ood_dl, postprocessor
                )
                
                pred = np.concatenate([id_pred, ood_pred])
                conf = np.concatenate([id_conf, ood_conf])
                conf1 = np.concatenate([id_conf1, ood_conf1])
                ddood = np.concatenate([id_ddood, ood_ddood])
                scood = np.concatenate([id_scood, ood_scood]) 

                if dataset_type == "scood":
                    label = scood
                elif dataset_type == "ddood":
                    label = ddood

                
                if i==0:
                    logging.info(f"Computing metrics on {id_name} + {ood_name} dataset...")
                    results = compute_all_metrics(conf, conf1, label, pred, output_dir)
                else:
                    results = compute_all_metrics(conf, conf1, label, pred, output_dir,verbose=False)
                # self._log_results(results, csv_path, dataset_name=ood_name)

                results_matrix.append(results)
            
        elif method == "each":
            results_matrix = []

            id_name = id_data_loader.dataset.name 

            logging.info(f"Performing inference on {id_name} dataset...")
            id_pred, id_conf, id_conf1, id_ddood, id_scood = self.inference(
                id_data_loader, postprocessor
            )
            
            # 遍历OOD数据集
            for i, ood_dl in enumerate(ood_data_loaders):
                ood_name = ood_dl.dataset.name

                logging.info(f"Performing inference on {ood_name} dataset...")
                ood_pred, ood_conf, ood_conf1, ood_ddood, ood_scood = self.inference(
                    ood_dl, postprocessor
                )
                # if ood_name == "tin":
                #     print('aaa')
                pred = np.concatenate([id_pred, ood_pred])
                conf = np.concatenate([id_conf, ood_conf])
                conf1 = np.concatenate([id_conf1, ood_conf1])
                ddood = np.concatenate([id_ddood, ood_ddood])
                scood = np.concatenate([id_scood, ood_scood]) 

                if dataset_type == "scood":
                    label = scood
                elif dataset_type == "ddood":
                    label = ddood

                logging.info(f"Computing metrics on {id_name} + {ood_name} dataset...")
                results = compute_all_metrics(conf, conf1, label, pred, output_dir)
                self._log_results(results, csv_path, dataset_name=ood_name)

                results_matrix.append(results)

            results_matrix = np.array(results_matrix)

            logging.info(f"Computing mean metrics...")
            results_mean = np.mean(results_matrix, axis=0)
            [fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, accuracy] = results_mean
            recall = 0.95
            logging.info(
            "FPR@{}: {:.2f}, AUROC: {:.2f}, AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(
                recall, 100 * fpr, 100 * auroc, 100 * aupr_in, 100 * aupr_out
                )
            )
            logging.info(
            "CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f}, ACC: {:.2f}".format(
                ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100, accuracy * 100
                )
            )
            self._log_results(results_mean, csv_path, dataset_name="mean")

        elif method == "full":
            data_loaders = [id_data_loader] + ood_data_loaders  #cifar10+[tin]
            pred_list, conf_list, conf1_list, ddood_list, scood_list = ([],[],[],[],[],)
            for i, test_loader in enumerate(data_loaders):
                name = test_loader.dataset.name
                logging.info(f"Performing inference on {name} dataset...")
                pred, conf, conf1, ddood, scood = self.inference(test_loader, postprocessor)
                pred_list.extend(pred)
                conf_list.extend(conf)
                conf1_list.extend(conf1)
                ddood_list.extend(ddood)
                scood_list.extend(scood)

            pred_list = np.array(pred_list)
            conf_list = np.array(conf_list)
            conf1_list=np.array(conf1_list)
            ddood_list = np.array(ddood_list).astype(int)
            scood_list = np.array(scood_list).astype(int)

            if dataset_type == "scood":
                label_list = scood_list
            elif dataset_type == "ddood":
                label_list = ddood_list

            logging.info(f"Computing metrics on combined dataset...")
            results = compute_all_metrics(conf_list, conf1_list, label_list, pred_list, output_dir)
            if csv_path:
                self._log_results(results, csv_path, dataset_name="full")

    def _log_results(self, results, csv_path, dataset_name=None):
        [fpr, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, accuracy] = results

        write_content = {
            "dataset": dataset_name,
            "FPR@95": "{:.2f}".format(100 * fpr),
            "AUROC": "{:.2f}".format(100 * auroc),
            "AUPR_IN": "{:.2f}".format(100 * aupr_in),
            "AUPR_OUT": "{:.2f}".format(100 * aupr_out),
            "CCR_4": "{:.2f}".format(100 * ccr_4),
            "CCR_3": "{:.2f}".format(100 * ccr_3),
            "CCR_2": "{:.2f}".format(100 * ccr_2),
            "CCR_1": "{:.2f}".format(100 * ccr_1),
            "ACC": "{:.2f}".format(100 * accuracy),
        }
        fieldnames = list(write_content.keys())
        # print(write_content, flush=True)

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)
