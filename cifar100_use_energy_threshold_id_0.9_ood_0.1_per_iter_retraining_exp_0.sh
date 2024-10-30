#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR1=$1
DATA_DIR=$2
seed=109

python train.py \
    --seed $seed \
    --config configs/train/cifar100_balanced_ft.yml \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --output_dir new_exp_results/cifar100_use_energy_threshold_id_0.9_ood_0.1_per_iter_retraining_seed_${seed}/cifar100 \
    --checkpoint /defaultShare/archive/pengzhimao/code/ood/ET-OOD-repeat/new_exp_results/zzz/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_120/cifar100/best.ckpt \
    --ood_mask_dir /defaultShare/archive/pengzhimao/code/ood/ET-OOD-repeat/new_exp_results/zzz/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_120/cifar100/final_ood_mask.npy \
    --id_mask_dir /defaultShare/archive/pengzhimao/code/ood/ET-OOD-repeat/new_exp_results/zzz/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_120/cifar100/final_id_mask.npy \
    --id_pseudo_label_dir /defaultShare/archive/pengzhimao/code/ood/ET-OOD-repeat/new_exp_results/zzz/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_120/cifar100/id_pseudo_label.npy \
    --use_balanced_fine_tuning \
    --use_backbone_fine_tuning \
    --all_retraining \

python test.py \
    --config configs/test/cifar100.yml \
    --checkpoint new_exp_results/cifar100_use_energy_threshold_id_0.9_ood_0.1_per_iter_retraining_seed_${seed}/cifar100/best.ckpt \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --csv_path new_exp_results/cifar100_use_energy_threshold_id_0.9_ood_0.1_per_iter_retraining_seed_${seed}/cifar100/results_msp.csv


