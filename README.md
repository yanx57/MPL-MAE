# MPL-MAE

This repository contains the official implementation of the paper
**"Mitigating Positional Leakage in 3D Masked Autoencoders for Robust Representation Learning"**.

> 🚧 **Code coming soon.** We are cleaning up the code and will release it
> here.

## Datasets

We evaluate MPL-MAE on a variety of point cloud datasets covering pre-training, classification, segmentation, registration, and reconstruction tasks.

- **ShapeNet** — used for self-supervised pre-training. Please follow the data preparation instructions from [PCP-MAE](https://github.com/aHapBean/PCP-MAE).
- **ModelNet40** — used for shape classification and few-shot learning. Please follow the data preparation from [PCP-MAE](https://github.com/aHapBean/PCP-MAE).
- **ScanObjectNN** — used to evaluate classification on real-world scanned objects (including the challenging `PB_T50_RS` / "hardest" variant). Please follow the data preparation from [PCP-MAE](https://github.com/aHapBean/PCP-MAE).
- **S3DIS** — used for indoor semantic segmentation. Please follow the data preparation from [PCP-MAE](https://github.com/aHapBean/PCP-MAE).
- **DCP** — used for the point cloud registration task. Please follow the instructions in the [DCP repository](https://github.com/WangYueFt/dcp/tree/master).
- **PCN** — used for the reconstruction / completion task. Please follow the instructions in the [PoinTr repository](https://github.com/yuxumin/PoinTr).

## Installation

### Conda environment

```bash
# Create environment
conda create -n pcpmae python=3.10 -y
conda activate pcpmae

# Install PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
# Alternatively:
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install required packages
pip install -r requirements.txt
```

### Install the extensions

```bash
# Chamfer Distance & EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Usage

### Pre-training (ShapeNet)

```bash
CUDA_VISIBLE_DEVICES=<GPU> python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name <output_file_name>
```

### Fine-tuning

**ScanObjectNN (hardest split):**

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name <output_file_name> \
    --ckpts <path/to/pre-trained/model> \
    --seed $RANDOM
```

**ModelNet40:**

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main.py \
    --config cfgs/finetune_modelnet.yaml \
    --finetune_model \
    --exp_name <output_file_name> \
    --ckpts <path/to/pre-trained/model> \
    --seed $RANDOM
```

**Voting on ModelNet40:**

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test \
    --config cfgs/finetune_modelnet.yaml \
    --exp_name <output_file_name> \
    --ckpts <path/to/best/fine-tuned/model> \
    --seed $RANDOM --vote
```

### Few-shot learning

```bash
CUDA_VISIBLE_DEVICES=<GPUs> python main.py \
    --config cfgs/fewshot.yaml --finetune_model \
    --ckpts <path/to/pre-trained/model> \
    --exp_name <output_file_name> \
    --way <5 or 10> --shot <10 or 20> --fold <0-9> \
    --seed $RANDOM
```

### Semantic segmentation (S3DIS)

```bash
cd semantic_segmentation
python main.py \
    --ckpts <path/to/pre-trained/model> \
    --root <path/to/data> \
    --learning_rate 0.0002 --epoch 60 \
    --gpu <gpu_id> --log_dir <log_dir>
```

### Registration (DCP)

```bash
cd registry
bash run.sh   # remember to modify the paths inside run.sh
```

### Reconstruction (PCN)

```bash
cd recon
bash run.sh   # remember to modify the paths inside run.sh
```

## Pre-trained Models & Experiment Logs

We provide pre-trained checkpoints and full experiment logs (training curves, evaluation outputs) for reproducing the results reported in the paper.

| Task | Dataset | Config | Checkpoint | Log |
| --- | --- | --- | --- | --- |
| Pre-training | ShapeNet | `cfgs/pretrain/base.yaml` | [ckpt-epoch-300.pth](ckpts/ckpt-epoch-300.pth) | [log](logs/pretrain.log) |
| Classification | ScanObjectNN (hardest) | `cfgs/finetune_scan_hardest.yaml` | coming soon | [log](logs/finetune_scan_hardest.log) |
| Classification | ModelNet40 | `cfgs/finetune_modelnet.yaml` | coming soon | [log](logs/finetune_modelnet.log) |
| Few-shot | ModelNet40 | `cfgs/fewshot.yaml` | coming soon | [log](logs/fewshot.log) |
| Segmentation | S3DIS | `semantic_segmentation/` | coming soon | [log](logs/s3dis.log) |
| Registration | DCP (ModelNet40) | `registry/` | coming soon | [log](logs/registration.log) |
| Reconstruction | PCN | `recon/` | coming soon | [log](logs/reconstruction.log) |

> Checkpoints will be released after the code cleanup. Log paths above point to files under `logs/` in this repository.

## Citation
