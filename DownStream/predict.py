import numpy as np
import torch
import cv2
import torch.nn as nn
from Model.unet import UNet
from utils.dataset import FundusSeg_Loader
from utils.eval_metrics import perform_metrics
import copy
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import os
target_domain = "idrid"
source_domain = "idrid"

#model_path='weight_files/withoutSSL/idrid25.pth'
model_path = 'weight_files/patch消融实验/idrid_0.3_3500e.pth'
if source_domain == "gendata":
    dataset_mean=[0.472, 0.297, 0.127]
    dataset_std=[0.297,0.202,0.129]

if source_domain == "drive":
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]
if source_domain == "idrid":
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]
if source_domain == "stare":
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]

if source_domain == "chase":
    dataset_mean=[0.4416, 0.1606, 0.0277]
    dataset_std=[0.3530,0.1407,0.0366]

if source_domain == "rim":
    dataset_mean=[0.3464, 0.1231, 0.0483]
    dataset_std=[0.1885,0.0983,0.0482]
if source_domain == "gs":
    dataset_mean=[0.3464, 0.1231, 0.0483]
    dataset_std=[0.1885,0.0983,0.0482]
if source_domain == "hei-med":
    dataset_mean=[0.3464, 0.1231, 0.0483]
    dataset_std=[0.1885,0.0983,0.0482]
if target_domain == "drive":
    test_data_path = "dataset/drive/test/"

if target_domain == "chase":
    test_data_path = "../dataset/chase_db1/test/"

if target_domain == "stare":
    test_data_path = "dataset/stare/test/"

if target_domain == "gendata":
    test_data_path = "../generate_vessel/gendata_new53/test/"

if target_domain == "rim":
    test_data_path = "dataset/rim/test/ROIs/"

if target_domain == "gs":
    test_data_path = "dataset/gs/test/ROIs/"
if target_domain == "hei-med":
    test_data_path = "dataset/hei-med/test/"
if target_domain == "idrid":
    test_data_path = "dataset/idrid/idrid_512/test/"
save_path='./result_image/rim/'

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, target_domain, dataset_mean, dataset_std)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader.dataset))
        # 选择设备CUDA
        device = torch.device('cuda')
        # 加载网络，图片单通道，分类为1。
        net = UNet(3, 1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        # 测试模式
        net.eval()
        pre_stack = []
        label_stack = []

        for image, label, filename, raw_height, raw_width in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            # image 的形状[batch_size, channel=1, height, width]
            pred = net(image)
            # Normalize to [0, 1]
            pred = torch.sigmoid(pred)
            # pred的形状 [batch_size, channel=1, height, weight]
            # raw_height, raw_height 是原始图像的高和宽
            pred  = pred[:,:,:raw_height,:raw_width]
            label = label[:,:,:raw_height,:raw_width]
            # 拿到pred的batch_size第一张图片
            pred = pred.cpu().numpy().astype(np.double)[0][0]
            label = label.cpu().numpy().astype(np.double)[0][0]
            pre_stack.append(pred)
            label_stack.append(label)
            # 保存图片
            # pred之前被归一化了(sigmoid)
            pred = pred * 255
            print(filename[0])
            os.makedirs(save_path, exist_ok=True)
            save_filename = save_path + filename[0] + '.png'
            cv2.imwrite(save_filename, pred)
            #print(f'{save_filename} done!')

        print('Evaluating...')
        label_stack = np.stack(label_stack, axis=0)
        pre_stack = np.stack(pre_stack, axis=0)
        label_stack = label_stack.reshape(-1)
        pre_stack = pre_stack.reshape(-1)

        precision, sen, spec, f1, acc, roc_auc, pr_auc = perform_metrics(pre_stack, label_stack)
        print(f'F1-score: {f1} PR_AUC: {pr_auc}')
