## DualScaleNet: A Self-Supervised Dual-Branch Contrastive Learning Framework for Enhanced Medical Image Segmentation
### Introduction 
This repository contains the implementation of DualScaleNet, a dual-branch contrastive learning framework designed for medical image segmentation. 
![123](https://github.com/meco66666/DualScaleNet/blob/main/DualScaleNet.png?raw=true)
### Self-supervised Preparation
Install PyTorch and DDR dataset.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
The DDR dataset is available for download at https://github.com/nkicsl/DDR-dataset. The dataset is organized as follows. To construct the self-supervised training set, images from the test set need to be manually merged with the training set.
```
DualScaleNet/
└── Dataset/
    └── ddr_757/
        ├── train/
        │   ├── img/          # Training images (e.g. 20170629163635747.jpg)
        │   └── label/       # Unused in pretraining
        ├── val/             # Validation set
        └── test/            # Test images to be merged with training set
```
The training process solely relies on the image data, without utilizing any label annotations.
### Self-Supervised Training
##### Default hyperparameters match those in our paper
##### Tested on NVIDIA RTX 3090 (24GB VRAM)
##### Single-GPU training supported
```python
python DualScaleNet/main.py
```
The script uses the default hyperparameters specified in the DualScaleNet paper.
### Downstream task evaluation
Freeze the weights/features trained by DualScaleNet and evaluate its performance on downstream tasks.
#### DataSet
We evaluate our method on five publicly available fundus image datasets:
- [DRIVE](https://drive.grand-challenge.org/) (Digital Retinal Images for Vessel Extraction)
- [STARE](https://cecas.clemson.edu/~ahoover/stare/) (STructured Analysis of the Retina)
- [RIM-ONE-r3](https://rimone.webs.ull.es/) (Retinal IMage ONline Examination)
- [Drishti-GS](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) (Drishti-Glaucoma Screening)
- [IDRiD](https://idrid.grand-challenge.org/) (Indian Diabetic Retinopathy Image Dataset)
```python
python DownStream/train.py
python DownStream/predict.py
```







