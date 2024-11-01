# Predictive Sample Assignment for Semantically Coherent Out-of-Distribution Detection
This repo contains code for our paper:“Predictive Sample Assignment for Semantically Coherent Out-of-Distribution Detection”.

# Abstract 

Semantically coherent out-of-distribution detection (SCOOD) is a recently proposed realistic  OOD detection setting: given labeled in-distribution (ID) data and mixed in-distribution and out-of-distribution unlabeled data as the training data, SCOOD aims to enable the trained model to accurately identify OOD samples in the testing data. Current SCOOD methods mainly adopt various clustering-based in-distribution sample filtering (IDF) strategies to select clean ID samples from unlabeled data, and take the remaining samples as auxiliary OOD data, which inevitably introduces a large number of noisy samples in training. To address the above issue, we propose a concise SCOOD framework based on predictive sample assignment (PSA). PSA includes a dual-threshold ternary sample assignment strategy based on the predictive energy score that can significantly improve the purity of the selected ID and OOD sample sets by assigning unconfident unlabeled data to an additional discard sample set, and a concept contrastive representation learning loss to further expand the distance between ID and OOD samples in the representation space to assist ID/OOD discrimination. In addition, we also introduce a retraining strategy to help the model fully fit the selected auxiliary ID/OOD samples. Experiments on two standard SCOOD benchmarks demonstrate that our approach outperforms the state-of-the-art methods by a significant margin.

# Running

### Dependencies

```
pip install -r requirements.txt
```

# SC-OOD Dataset

The SC-OOD dataset introduced in the paper can be downloaded here: [![gdrive](https://img.shields.io/badge/SCOOD%20dataset-google%20drive-f39f37)](https://drive.google.com/file/d/1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view?usp=sharing)1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view

Our codebase accesses the dataset from the root directory in a folder named `data/` by default, i.e.

```
├── ...
├── data
│   ├── images
│   └── imglist
├── scood
├── test.py
├── train.py
├── ...
```

# Training

The entry point for training is the `train.py` script. The hyperparameters for each experiment is specified by a `.yml` configuration file (examples given in [`configs/train/`](configs/train/)).

All experiment artifacts are saved in the specified `args.output_dir` directory.

### PSA Selection Training

```
bash cifar10_threshold_training.sh
```

At the same time, three additional files (final_ood_mask.npy, final_id_mask.npy, id_pseudo_label.npy) for retraining are saved. 

### PSA Retraining and Testing 

```
bash cifar10_retraining.sh
```

The evaluation results are saved in a `.csv` file as specified.  

# Acknowledgements 

The code base is largely built on this repo:https://github.com/jingkang50/ICCV21_SCOOD?tab=readme-ov-file

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

