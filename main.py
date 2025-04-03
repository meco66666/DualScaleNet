import os
import copy
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
backbone_dir = os.path.join(current_dir, '..', 'backbone')
sys.path.append(backbone_dir)
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from resnet.resnet import resnet18 as resnet
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from transform.Nine_Global_npatch_transform import AugmentedImageSplitter
from Fundus_DataSet.Nine_Global_npatch_Fundus_DataSet import Fundus_Dataset
from Loss.Nine_Global_npatch_loss import custom_similarity_loss

class Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
    def forward_projection(self, x):
        _, _, _, _, f = self.encoder(x)
        z = self.pool(f).flatten(start_dim=1)
        z = self.projection_head(z)
        return z
    def forward_prediction(self,x):
        z = self.forward_projection(x)
        p = self.prediction_head(z)
        return p
        
model = Model(resnet)
device = "cuda"
model.to(device)
transform = AugmentedImageSplitter()
dataset = Fundus_Dataset("../Dataset/ddr_757/train/img/", transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_interval = 500
model_save_path = 'weight_files/Global+nine/Global_npatch/5patch_0.6GP_64b_Adam_{epoch}.pth'
save_epochs = []
losses = []
epochs = 5000
visualizations_dir = os.path.join(current_dir, 'visualizations')
print("Starting Training")
local_losses = []
global_losses = []
total_losses = []
lowest_loss = float('inf')
lowest_epoch = -1
loss_image_path = "Loss_Image/Global_npatch/losses_curve_5patch_0.6GP_curve.png"
loss_file_path = "Loss_Image/Global_npatch/losses_5patch_0.6GP.txt"

for epoch in range(epochs):
    total_local_loss = 0
    total_global_loss = 0
    total_loss = 0
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        Local_patches, Global_patches, Global_distances = batch
        # Local
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9 = Local_patches
        x_vars = [x.permute(0, 3, 1, 2).to(dtype=torch.float).to(device) for x in [x1, x2, x3, x4, x5, x6, x7, x8, x9]]
        y_vars = [y.permute(0, 3, 1, 2).to(dtype=torch.float).to(device) for y in [y1, y2, y3, y4, y5, y6, y7, y8, y9]]
        zx_vars = [model.forward_prediction(x) for x in x_vars]
        zy_vars = [model.forward_projection(y) for y in y_vars]
        # Global
        positive_1_container = []
        positive_2_container = []
        negative_1_container = []
        negative_2_container = []
        patch_pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4)
        ]
        for b in range(64):
            x1 = Global_patches[0][b]
            x2 = Global_patches[1][b]
            x3 = Global_patches[2][b]
            x4 = Global_patches[3][b]
            x5 = Global_patches[4][b]
            x1 = x1.permute(2, 0, 1).to(dtype=torch.float).to(device)
            x2 = x2.permute(2, 0, 1).to(dtype=torch.float).to(device)
            x3 = x3.permute(2, 0, 1).to(dtype=torch.float).to(device)
            x4 = x4.permute(2, 0, 1).to(dtype=torch.float).to(device)
            x5 = x5.permute(2, 0, 1).to(dtype=torch.float).to(device)
            b_patches = [x1, x2, x3, x4, x5]
            all_distances = [dist[b] for dist in Global_distances]
            all_distances = torch.tensor(all_distances)
            min_idx = torch.argmin(all_distances).item()
            max_idx = torch.argmax(all_distances).item()
            min_patch_pair = patch_pairs[min_idx]
            max_patch_pair = patch_pairs[max_idx]
            min_i, min_j = min_patch_pair
            max_i, max_j = max_patch_pair
            positive_1_container.append(b_patches[min_i])
            positive_2_container.append(b_patches[min_j])
            negative_1_container.append(b_patches[max_i])
            negative_2_container.append(b_patches[max_j])
        positive_1_container_tensor = torch.stack(positive_1_container)
        positive_2_container_tensor = torch.stack(positive_2_container)
        negative_1_container_tensor = torch.stack(negative_1_container)
        negative_2_container_tensor = torch.stack(negative_2_container)
        
        positive_1_container_encoder = model.forward_prediction(positive_1_container_tensor)
        positive_2_container_encoder = model.forward_projection(positive_2_container_tensor)
        negative_1_container_encoder = model.forward_prediction(negative_1_container_tensor)
        negative_2_container_encoder = model.forward_projection(negative_2_container_tensor)
        Local_positive_loss,Local_negative_loss,Global_positive_loss,Global_negative_loss = custom_similarity_loss(zx_vars,zy_vars,positive_1_container_encoder,positive_2_container_encoder,negative_1_container_encoder,negative_2_container_encoder)
        
        local_loss = 0.7 * Local_positive_loss + 0.3 * Local_negative_loss
        global_loss = 0.6 * Global_positive_loss + 0.4 * Global_negative_loss
        loss = local_loss + global_loss
        
        total_local_loss += local_loss.detach().item()
        total_global_loss += global_loss.detach().item()
        total_loss += loss.detach().item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    avg_local_loss = total_local_loss / batch_count
    avg_global_loss = total_global_loss / batch_count
    avg_total_loss = total_loss / batch_count

    local_losses.append(avg_local_loss)
    global_losses.append(avg_global_loss)
    total_losses.append(avg_total_loss)
    if (epoch+1) % save_interval == 0:
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        print(f"Epoch: {epoch+1}, avg loss: {avg_loss:.5f}")
        model_path = model_save_path.format(epoch=epoch+1)
        torch.save(model.state_dict(), model_path)
        print(f"Saving model to {model_path}")
        save_epochs.append(epoch + 1)
    if avg_total_loss < lowest_loss:
        lowest_loss = avg_total_loss
        lowest_epoch = epoch + 1

# 保存损失曲线图
plt.figure(figsize=(10, 6))
plt.plot(local_losses, label="Local Loss", color="blue")
plt.plot(global_losses, label="Global Loss", color="green")
plt.plot(total_losses, label="Total Loss", color="red")
plt.scatter([lowest_epoch - 1], [lowest_loss], color="black", label="Lowest Loss", zorder=5)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(loss_image_path)
plt.show()

# 保存损失数据到文
with open(loss_file_path, "w") as f:
    for epoch, (local, global_, total) in enumerate(zip(local_losses, global_losses, total_losses), start=1):
        f.write(f"Epoch {epoch}:\n")
        f.write(f"  Local Loss:  {local:.5f}\n")
        f.write(f"  Global Loss: {global_:.5f}\n")
        f.write(f"  Total Loss:  {total:.5f}\n")
    f.write(f"\nLowest Total Loss: {lowest_loss:.5f} at Epoch {lowest_epoch}\n")

print(f"Loss curve saved to '{loss_image_path}' and losses recorded in '{loss_file_path}'.")
        
        
        
        
        
        
        
        
        
        
        
