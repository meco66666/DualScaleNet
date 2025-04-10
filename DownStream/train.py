from Model.unet import UNet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import time
import os
import matplotlib.pyplot as plt

# 初始化损失列表
val_losses = []
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
dataset_name = "idrid"
if dataset_name == "idrid":
    train_data_path = "dataset/idrid/idrid_512/train/"
    #train_data_path = "dataset/idrid/idrid_512/train_25/"
    valid_data_path = "dataset/idrid/idrid_512/test/"
    N_epochs = 300
    # lr_decay_step = [100]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]
if dataset_name == "drive":
    train_data_path = "dataset/drive/train/"
    valid_data_path = "dataset/drive/test/"
    N_epochs = 280
    # lr_decay_step = [100]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]


if dataset_name == "stare":
    train_data_path = "dataset/stare/train/"
    valid_data_path = "dataset/stare/test/"
    N_epochs = 300      #epoch = 400
    # lr_decay_step = [100]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]

if dataset_name == "rim":
    train_data_path = "dataset/rim/train/ROIs/"
    #train_data_path = "dataset/rim/train/"
    valid_data_path = "dataset/rim/test/ROIs/"
    N_epochs = 300
    # lr_decay_step = [100]
    lr_init = 0.0001
    batch_size = 2
    test_epoch = 5
    dataset_mean=[0.3464, 0.1231, 0.0483]
    dataset_std=[0.1885,0.0983,0.0482]
if dataset_name == "gs":
    train_data_path = "dataset/gs/train/ROIs/"
    #train_data_path = "dataset/gs/train_25/"
    valid_data_path = "dataset/gs/test/ROIs/"
    N_epochs = 300
    # lr_decay_step = [100]
    lr_init = 0.0001
    batch_size = 2
    test_epoch = 5
    dataset_mean=[0.3464, 0.1231, 0.0483]
    dataset_std=[0.1885,0.0983,0.0482]
if dataset_name == "hei-med":
    train_data_path = "dataset/hei-med/train/"
    valid_data_path = "dataset/hei-med/test/"
    N_epochs = 400
    # lr_decay_step = [100]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]

def save_losses_to_file(val_losses, file_path):
    # 找到最低验证损失及对应的 epoch
    min_val_loss = min(val_losses)
    min_epoch = val_losses.index(min_val_loss) + 1

    with open(file_path, 'w') as f:
        f.write("Epoch\tValidation Loss\n")
        for epoch, val_loss in enumerate(val_losses, 1):
            f.write(f"{epoch}\t{val_loss:.6f}\n")
        f.write("\n")
        f.write(f"Minimum Validation Loss: {min_val_loss:.6f} at Epoch {min_epoch}\n")


def train_net(net, device, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    # 加载训练集
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name, dataset_mean, dataset_std)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name, dataset_mean, dataset_std)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Train images: %s' % len(train_loader.dataset))
    print('Valid images: %s' % len(valid_loader.dataset))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    # 学习率调度器，milestones=[30,60]表示在第30和60的epoch的时候将 学习率乘以gamma=0.1
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    # 训练epochs次
    # 求最小值，所以初始化为正无穷
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        # 训练模式
        net.train()
        train_loss = 0
        for i, (image, label, filename, raw_height, raw_width) in enumerate(train_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

        # Validation
        # epoch != test_epoch
        # 每训练5轮进行一次评估
        # if ((epoch+1) % test_epoch == 0):
        with torch.no_grad():
            net.eval()
            val_loss = 0
            for i, (image, label, filename, raw_height, raw_width) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                val_loss = val_loss + loss.item()

            # net.state_dict()就是用来保存模型参数的
            if val_loss < best_loss:
                best_loss = val_loss
                #torch.save(net.state_dict(), 'weight_files/simsiam/'+dataset_name+'_757images_64b_Adam5000.pth')
                torch.save(net.state_dict(), 'weight_files/patch消融实验/' + dataset_name + '_0.3_3500e.pth')
                #torch.save(net.state_dict(), 'weight_files/withoutSSL/'+dataset_name+'.pth')
                print('saving model............................................')
        
            print('Loss/valid', val_loss / (i + 1))
            val_loss = val_loss / (i+1)
            val_losses.append(val_loss)
            sys.stdout.flush()
        # scheduler.step()

if __name__ == "__main__":
    device = torch.device('cuda')
    net = UNet(3, 1)
    net.load_state_dict(torch.load('../examples/weight_files/Local/0.3pos_3500e.pth'), strict=False)
    net.to(device=device)
    train_net(net, device)
    # 绘制训练损失和验证损失曲线
    #loss_file_path = "Loss_Image/WithoutSSL/idrid_losses.txt"
    #loss_file_path = "Loss_Image/nine+simsiam/idrid_4000e_losses.txt"
    #save_losses_to_file(val_losses, loss_file_path)
