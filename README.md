# QuantV2X: Cooperative Perception with PointPillarCluster

This repository contains the implementation of **PointPillarCluster**, a cooperative perception model for multi-agent 3D object detection. The model combines PointPillar's efficient detection with cluster-based fusion for robust vehicle-infrastructure cooperation.

## Quick Links

- **[Configuration Reference](opencood/hypes_yaml/exemplar.yaml)** - Detailed explanation of all configuration options

## Features

- Pure PyTorch implementation (no mmdet3d CUDA extensions required)
- Cluster-based fusion with pose graph optimization
- Optimized for DAIR-V2X-C dataset
- ~30MB model size, suitable for edge deployment

## Quick Start

### 1. Installation

```bash
# Create environment
conda create -n quantv2x python=3.8 pytorch==1.12.0 torchvision==0.13.0 \
    cudatoolkit=11.6 -c pytorch -c conda-forge
conda activate quantv2x

# Install dependencies
pip install -r requirements.txt
python setup.py develop

# Install spconv
pip install spconv-cu116  # Match your CUDA version

# Compile CUDA extensions
python opencood/utils/setup.py build_ext --inplace

# Install additional dependencies
pip install g2o-python
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

### 2. Dataset Preparation

Download **DAIR-V2X-C** dataset:
- Official page: https://thudair.baai.ac.cn/index
- **Important**: Use [complemented annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/)

Organize dataset as:
```
QuantV2X/
└── dataset/
    └── my_dair_v2x/
        └── v2x_c/
            └── cooperative-vehicle-infrastructure/
```

### 3. Training

```bash
# Single GPU training
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pointpillar_cluster.yaml

# Multi-GPU training (recommended)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --use_env opencood/tools/train_ddp.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pointpillar_cluster.yaml
```

### 4. Evaluation

```bash
python opencood/tools/inference.py \
    --model_dir opencood/logs/PointPillarCluster_DAIR_<timestamp> \
    --fusion_method intermediate
```

## Performance

**DAIR-V2X-C Results:**
- AP@0.5: 65.3%
- AP@0.7: 51.8%

## Model Architecture

```
Point Cloud → PillarVFE → BEV Backbone → ClusterFusion → Detection Head → 3D Boxes
```

**Key Components:**
- **PillarVFE**: Efficient pillar-based feature extraction
- **ClusterFusion**: Cluster-based multi-agent fusion with pose optimization
- **Detection Head**: Anchor-based 3D object detection

## Documentation

For detailed instructions, see:
- [PointCluster Training Guide](POINTCLUSTER_TRAINING.md) - Complete training tutorial
- [Configuration Guide](opencood/hypes_yaml/exemplar.yaml) - All configuration options
- Model code: [point_pillar_cluster.py](opencood/models/point_pillar_cluster.py)
- Fusion code: [cluster_fusion.py](opencood/models/fuse_modules/cluster_fusion.py)

## Repository Structure

```
QuantV2X/
├── opencood/
│   ├── models/
│   │   ├── point_pillar_cluster.py       # Main model
│   │   ├── fuse_modules/
│   │   │   ├── cluster_fusion.py         # Cluster fusion
│   │   │   └── mmdet3d_ops_standalone.py # Pure PyTorch ops
│   │   └── sub_modules/
│   │       ├── pillar_vfe.py             # Pillar feature encoder
│   │       ├── cluster_align.py          # Pose refinement
│   │       └── ...
│   ├── hypes_yaml/
│   │   └── dairv2x/LiDAROnly/
│   │       └── lidar_pointpillar_cluster.yaml  # Config
│   ├── tools/
│   │   ├── train.py                      # Training script
│   │   ├── train_ddp.py                  # Multi-GPU training
│   │   └── inference.py                  # Evaluation
│   ├── data_utils/                       # Dataset loaders
│   └── loss/
│       └── point_pillar_loss.py          # Loss function
└── dataset/                               # Datasets (not in repo)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhao2025quantv2x,
 title={QuantV2X: A Fully Quantized Multi-Agent System for Cooperative Perception},
 author={Zhao, Seth Z and Zhang, Huizhi and Li, Zhaowei and Peng, Juntong and Chui, Anthony and Zhou, Zewei and Meng, Zonglin and Xiang, Hao and Huang, Zhiyu and Wang, Fujia and others},
 journal={arXiv preprint arXiv:2509.03704},
 year={2025}
}

@inproceedings{ding2025point,
title={Point Cluster: A Compact Message Unit for Communication-Efficient Collaborative Perception},
author={Zihan Ding and Jiahui Fu and Si Liu and Hongyu Li and Siheng Chen and Hongsheng Li and Shifeng Zhang and Xu Zhou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=54XlM8Clkg}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This project builds upon:
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) - Cooperative detection framework
- [PointPillars](https://arxiv.org/abs/1812.05784) - Efficient 3D object detection
- [DAIR-V2X](https://thudair.baai.ac.cn/) - Vehicle-infrastructure dataset
