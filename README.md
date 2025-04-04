## DualScaleNet: A Self-Supervised Dual-Branch Contrastive Learning Framework for Enhanced Medical Image Segmentation
This repository contains the implementation of DualScaleNet, a dual-branch contrastive learning framework designed for medical image segmentation. 
![123](https://github.com/meco66666/DualScaleNet/blob/main/12.png?raw=true)
### Preparation
Install PyTorch and DDR dataset.
The DDR dataset is available for download at https://github.com/nkicsl/DDR-dataset. The dataset is organized as follows. To construct the self-supervised training set, images from the test set need to be manually merged with the training set.
```
DualScaleNet/Dataset/ddr_757
    train/
        img/
            20170629163635747.jpg
            ...
        label/
    val/
    test/
        img/
            20170627170651362.jpg
            ...
```
The training process solely relies on the image data, without utilizing any label annotations.
### Self-Supervised Training
The implementation supports single-GPU training and has been empirically validated via self-supervised pretraining on an NVIDIA RTX 3090.
```python
python DualScaleNet/main.py
```
The script uses the default hyperparameters specified in the DualScaleNet paper.
