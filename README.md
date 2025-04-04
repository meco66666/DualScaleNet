# DualScaleNet: A Self-Supervised Dual-Branch Contrastive Learning Framework for Enhanced Medical Image Segmentation
This repository contains the implementation of DualScaleNet, a dual-branch contrastive learning framework designed for medical image segmentation. 
![123](https://github.com/meco66666/DualScaleNet/blob/main/12.png?raw=true)
### Preparation
Install PyTorch and DDR dataset following the https://github.com/nkicsl/DDR-dataset
### Self-Supervised Training
The implementation supports single-GPU training and has been empirically validated via self-supervised pretraining on an NVIDIA RTX 3090.
'''python
python DualScaleNet/main.py
'''
