#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seed=120

python train.py \
    --config configs/train/cifar100_ET.yml \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --output_dir new_exp_results/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_${seed}/cifar100 \
    --seed $seed \
    --use_threshold_training \
    

python test.py \
    --config configs/test/cifar100.yml \
    --checkpoint new_exp_results/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_${seed}/cifar100/best.ckpt \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --csv_path new_exp_results/use_energy_threshold_training_warmup_30_id_0.9_ood_0.1_per_iter_seed_${seed}/cifar100/results_msp.csv
