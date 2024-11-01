#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


seed=287
python train.py \
    --config configs/train/cifar10_ET.yml \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --output_dir new_exp_results/use_energy_threshold_training/cifar10 \
    --seed $seed \
    --use_threshold_training \
    

python test.py \
    --config configs/test/cifar10.yml \
    --checkpoint new_exp_results/use_energy_threshold_training/cifar10/best.ckpt \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --csv_path new_exp_results/use_energy_threshold_training/cifar10/results_msp.csv

