# MV<sub>Hybrid</sub>: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models

Official implementation of the paper accepted at MICCAI COMPAYL 2025 Workshop.

[![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/pdf?id=vd1xqJLW4X)

## Abstract

Spatial transcriptomics reveals gene expression patterns within tissue context, enabling precision oncology applications such as treatment response prediction, but its high cost and technical complexity limit clinical adoption. Predicting spatial gene expression (biomarkers) from routine histopathology images offers a practical alternative, yet current vision foundation models (VFMs) in pathology based on Vision Transformer (ViT) backbones perform below clinical standards. Given that VFMs are trained on millions of diverse whole slide images, we hypothesize that architectural innovations beyond ViTs may better capture the low-frequency, subtle morphological patterns correlating with molecular phenotypes. By demonstrating that state space models initialized with negative real eigenvalues exhibit strong low-frequency bias, we introduce MV<sub>Hybrid</sub>, a hybrid backbone architecture combining state space models (SSMs) with ViT. We compare five other different backbone architectures for pathology VFMs, all pretrained on identical colorectal cancer datasets using the DINOv2 self-supervised learning method. We evaluate all pretrained models using both random split and leave-one-study-out (LOSO) settings of the same biomarker dataset. In LOSO evaluation, MV<sub>Hybrid</sub> achieves 57% higher correlation than the best-performing ViT and shows 43% smaller performance degradation compared to random split in gene expression prediction, demonstrating superior performance and robustness, respectively. Furthermore, MV<sub>Hybrid</sub> shows equal or better downstream performance in classification, patch retrieval, and survival prediction tasks compared to that of ViT, showing its promise as a next-generation pathology VFM backbone.

## Installation

Create a conda environment and install the required packages:

```bash
conda create -n mambavision python==3.11
conda activate mambavision

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 xformers --index-url https://download.pytorch.org/whl/cu121
conda install nvidia/label/cuda-12.1.0::cuda-nvcc 
pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 timm==1.0.9 einops transformers
pip install fvcore submitit omegaconf
```

## Training

The training of DINOv2 was performed on a SLURM cluster with multi-node multi-GPU (Fully Sharded Data Parallel - FSDP) environment setup.

To train MV<sub>Hybrid</sub>, submit the SLURM job using:
```bash
sbatch dino/Train_MVHybrid_DINOv2_SLURM.sh
```

## Dataset

### Dataset Source
Whole-slide images (WSI) downloaded from:
- [HunCRC](https://doi.org/10.5281/zenodo.8274948)
- [IMP-CRS2024](https://doi.org/10.5281/zenodo.11092840)

### Data Processing
WSI patches were extracted using [CLAM](https://github.com/mahmoodlab/CLAM)'s patching function with biopsy preset.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{
cho2025mvhybrid,
title={$MV_{Hybrid}$: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models},
author={Won June Cho and Hongjun Yoon and Daeky Jeong and Hyeongyeol Lim and Yosep Chong},
booktitle={MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL)},
year={2025},
url={https://openreview.net/forum?id=vd1xqJLW4X}
}
```

## Acknowledgments

This repository builds upon and modifies code from:
- **[DINOv2](https://github.com/facebookresearch/dinov2)**: We adapted the DINOv2 self-supervised learning framework to work with our hybrid architectures.
- **[MambaVision](https://github.com/NVlabs/MambaVision)**: We integrated and modified the MambaVision architecture to enable compatibility with DINOv2 training.

The implementation required significant modifications to enable seamless integration between these two frameworks, allowing DINOv2 to train MambaVision-based architectures and vice versa.