import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import random
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import pdb
import logging

from scood.data import get_dataloader, get_ext_dataloader
from scood.evaluation import Evaluator
from scood.postprocessors import get_postprocessor
from scood.networks import ResNet18
from scood.trainers import get_ETtrainer, ETtrainer
from scood.utils import load_yaml, setup_logger

from utils_sampler import source_import, get_value



def main(args, config):

    seed = config["dataset"]["seed"]
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # expt_output_dir = (output_dir / config["name"]).mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config_save_path = output_dir / "config.yml"
    shutil.copy(config_path, config_save_path)

    # setup_logger(str(output_dir))
    logging.basicConfig(filename=str(output_dir)+'/log.txt', level=logging.INFO)
    
    logging.info("seed: {}".format(seed))
    benchmark = config["dataset"]["labeled"]
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100" or "ima100":
        num_classes = 100

    # Init Datasets ############################################################
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    set_seed(seed)
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
    )

    labeled_train_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="train",
        batch_size=config["dataset"]["labeled_batch_size"],
        shuffle=True,
        num_workers=args.prefetch
    )

    if config['dataset']['unlabeled'] == "none":
        unlabeled_train_loader = None
    else:
        unlabeled_train_loader = get_dataloader_default(
            name=config["dataset"]["unlabeled"],
            stage="train",
            batch_size=config["dataset"]["unlabeled_batch_size"],
            shuffle=True,
            num_workers=args.prefetch
        )
    
    set_seed(seed)
    get_dataloader_ext = partial(
        get_ext_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
    )

    labeled_aug_loader = get_dataloader_ext(
        name=config["dataset"]["labeled"],
        stage="train",
        batch_size=config["dataset"]["labeled_batch_size"],
        shuffle=True,
        num_workers=args.prefetch
    )

    
    if args.use_balanced_fine_tuning:
        
        sampler_defs = config['sampler']
        
        if sampler_defs:
            
            if sampler_defs['type'] == 'ClassAwareSampler':
                
                sampler_dic = {
                    'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                    'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
                }
            elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                        'ClassPrioritySampler']:
                sampler_dic = {
                    'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                    'params': {k: v for k, v in sampler_defs.items() \
                            if k not in ['type', 'def_file']}
                }
        else:
            sampler_dic = None
         
        
        if args.all_retraining:
            
            sampler_dic = None
         

    if config['dataset']['unlabeled'] == "none":
        unlabeled_aug_loader = None
    else:
        
        if args.use_balanced_fine_tuning:
            
            ood_mask = np.load(args.ood_mask_dir)
            
            id_mask = np.load(args.id_mask_dir)
            
            id_pseudo_label = np.load(args.id_pseudo_label_dir)
            
            if sampler_dic is None:
                shuffle = True
            else:
                shuffle = False 
            
            unlabeled_aug_loader = get_dataloader_ext(
                name=config["dataset"]["unlabeled"],
                stage="train",
                batch_size=config["dataset"]["unlabeled_batch_size"],
                shuffle=shuffle,
                num_workers=args.prefetch,
                ood_mask = ood_mask,
                id_mask = id_mask,
                id_pseudo_label = id_pseudo_label,
                sampler_dic = sampler_dic
            )

        else:
            unlabeled_aug_loader = get_dataloader_ext(
                name=config["dataset"]["unlabeled"],
                stage="train",
                batch_size=config["dataset"]["unlabeled_batch_size"],
                shuffle=True,
                num_workers=args.prefetch
            )

        unlabeled_aug_loader_mask = get_dataloader_default(
            name=config["dataset"]["unlabeled"],
            stage="train",
            batch_size=config["dataset"]["unlabeled_batch_size"],
            shuffle=False,
            num_workers=args.prefetch
        )


    test_id_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="test",
        batch_size=config["dataset"]["test_batch_size"],
        shuffle=False,
        num_workers=args.prefetch
    )

    test_ood_loader_list = []
    for name in config["dataset"]["test_ood"]:
        test_ood_loader = get_dataloader_default(
            name=name,
            stage="test",
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=False,
            num_workers=args.prefetch
        )
        test_ood_loader_list.append(test_ood_loader)


    try:
        num_clusters = config['trainer_args']['num_clusters']
    except KeyError:
        num_clusters = 0

    set_seed(seed)
    net = ResNet18(num_classes=num_classes, dim_aux=num_clusters)
    
    if args.use_balanced_fine_tuning:
        if not args.random_init_paras:
            checkpoint = args.checkpoint
            if checkpoint:
                net.load_state_dict(torch.load(checkpoint), strict=False)
                if args.use_fc_fine_tuning:
                    for param_name, param in net.named_parameters():
                            # Freeze all parameters except self attention parameters
                            if 'selfatt' not in param_name and 'fc' not in param_name:
                                param.requires_grad = False
        else:
            print('use random_init_paras')

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        set_seed(seed)
        net.cuda()


    # cudnn.benchmark = True 
    
    # id_threshold = 0 # np.load(args.id_threshold_dir)
    # ood_threshold = 0 # np.load(args.ood_threshold_dir)
    use_threshold_training = args.use_threshold_training
    use_balanced_fine_tuning = args.use_balanced_fine_tuning

    
    set_seed(seed)
    trainer = get_ETtrainer(net,
        labeled_train_loader,
        unlabeled_train_loader,
        labeled_aug_loader,
        unlabeled_aug_loader,
        unlabeled_aug_loader_mask,
        use_threshold_training,
        use_balanced_fine_tuning,
        config['lamda'], 
        config['contra_loss_type'],
        config['optim_args'],
        config['trainer_args']
    )

    # Start Training ###########################################################
    set_seed(seed)
    evaluator = Evaluator(net)

    output_dir = Path(args.output_dir)
    

    begin_epoch = time.time()
    best_accuracy = 0.0
    
    for epoch in range(0, config["optim_args"]["epochs"]):
        
        train_metrics = trainer.train_epoch(epoch, output_dir)

        
        classification_metrics = evaluator.eval_classification(test_id_loader)
        postprocess_args = config["postprocess_args"] if config["postprocess_args"] else {}
        postprocessor = get_postprocessor(config["postprocess"], **postprocess_args)
        evaluator.eval_ood(
            test_id_loader,
            test_ood_loader_list,
            postprocessor=postprocessor,
            method="sel",
            dataset_type="scood",
            output_dir = output_dir,
        )
        
        # Save model
        torch.save(net.state_dict(), output_dir / f"epoch_{epoch}.ckpt")
        if not args.save_all_model:
            # Let us not waste space and delete the previous model
            prev_path = output_dir / f"epoch_{epoch - 1}.ckpt"
            prev_path.unlink(missing_ok=True)

        # save best result
        if classification_metrics["test_accuracy"] >= best_accuracy:
            torch.save(net.state_dict(), output_dir / f"best.ckpt")

            best_accuracy = classification_metrics["test_accuracy"]

        logging.info(
            "Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc {:.2f}".format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                train_metrics["train_loss"],
                classification_metrics["test_loss"],
                100.0 * classification_metrics["test_accuracy"],
            ),
            # flush=True,
        )
    print('Training Completed!')


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        help="path to config file",
        default="configs/train/cifar10_ET.yml",
    )
    
    parser.add_argument(
        "--checkpoint",
        help="specify path to checkpoint if loading from pre-trained model",
        # default="3.1/pretrain.ckpt",
    )
    
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        default="data",
    )
    
    parser.add_argument(
        "--output_dir",
        help="directory to save experiment artifacts",
        default="output/cifar10",
    )
    
    parser.add_argument(
        "--save_all_model",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--id_threshold_dir",
        help="directory to save experiment artifacts",
        default="output/cifar10",
    )

    
    parser.add_argument(
        "--ood_threshold_dir",
        help="directory to save experiment artifacts",
        default="output/cifar10",
    )

    
    parser.add_argument(
        "--use_threshold_training",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--use_balanced_fine_tuning",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--use_fc_fine_tuning",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--use_backbone_fine_tuning",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--ood_mask_dir",
        help="directory to save experiment artifacts",
        default="new_exp_results/use_energy_threshold_training_warmup_30_id_0.95_ood_0.3_exp_0/cifar100/final_ood_mask.npy",
    )

    
    parser.add_argument(
        "--id_mask_dir",
        help="directory to save experiment artifacts",
        default="new_exp_results/use_energy_threshold_training_warmup_30_id_0.95_ood_0.3_exp_0/cifar100/final_id_mask.npy",
    )

    
    parser.add_argument(
        "--id_pseudo_label_dir",
        help="directory to save experiment artifacts",
        default="new_exp_results/use_energy_threshold_training_warmup_30_id_0.95_ood_0.3_exp_0/cifar100/id_pseudo_label.npy",
    )

    
    parser.add_argument(
        "--all_retraining",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument(
        "--random_init_paras",
        action="store_true",
        help="whether to save all model checkpoints",
    )

    
    parser.add_argument("--seed", type=int, help="number of seed to use")

    
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=16, help="pre-fetching threads.")
    
    
    args = parser.parse_args()

    
    config = load_yaml(args.config)

    

    
    main(args, config)
