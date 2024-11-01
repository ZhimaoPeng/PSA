export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR1=$1
DATA_DIR=$2



seed=287
python train.py \
    --seed $seed \
    --config configs/train/cifar10_balanced_ft.yml \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --output_dir new_exp_results/cifar10_use_energy_threshold_retraining/cifar10 \
    --checkpoint new_exp_results/use_energy_threshold_training/cifar10/best.ckpt \
    --ood_mask_dir new_exp_results/use_energy_threshold_training/cifar10/final_ood_mask.npy \
    --id_mask_dir new_exp_results/use_energy_threshold_training/cifar10/final_id_mask.npy \
    --id_pseudo_label_dir new_exp_results/use_energy_threshold_training/cifar10/id_pseudo_label.npy \
    --use_balanced_fine_tuning \
    --use_backbone_fine_tuning \
    --all_retraining \

python test.py \
    --config configs/test/cifar10.yml \
    --checkpoint new_exp_results/cifar10_use_energy_threshold_retraining/cifar10/best.ckpt \
    --data_dir /defaultShare/archive/pengzhimao/code/ET-OOD-NEW/data \
    --csv_path new_exp_results/cifar10_use_energy_threshold_retraining/cifar10/results_msp.csv
