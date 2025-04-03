import os
import copy
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
backbone_dir = os.path.join(current_dir, '..', 'backbone')
sys.path.append(backbone_dir)
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from resnet.resnet import resnet18 as resnet
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from transform.nine_transform_global import AugmentedImageSplitter
from Fundus_dataset_nine_global import Fundus_Dataset
from Loss.nine_simsiam_loss import custom_similarity_loss
import torch.nn.functional as F
def simsiam_loss(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -torch.mean(torch.sum(p * z, dim=1))
class Nine(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.local_projection_head = SimSiamProjectionHead(512, 512, 128)
        self.local_prediction_head = SimSiamPredictionHead(128, 64, 128)
        
        self.global_projection_head = SimSiamProjectionHead(512, 512, 128)
        #512-512-256 256-128-256
        self.global_prediction_head = SimSiamPredictionHead(128, 64, 128)
    def forward_projection(self, x):
        _, _, _, _, f = self.encoder(x)
        z = self.pool(f).flatten(start_dim=1)
        z = self.local_projection_head(z)
        return z
    def forward_prediction(self,x):
        z = self.forward_projection(x)
        p = self.local_prediction_head(z)
        return p

    def global_forward_projection(self, x):
        _, _, _, _, f = self.encoder(x)
        z = self.pool(f).flatten(start_dim=1)
        z = self.global_projection_head(z)
        return z
    def global_forward_prediction(self,x):
        z = self.global_forward_projection(x)
        p = self.global_prediction_head(z)
        return p
    def simsiam_forward(self, x1, x2):
        p1 = self.global_forward_prediction(x1)
        z2 = self.global_forward_projection(x2).detach()
        p2 = self.global_forward_prediction(x2)
        z1 = self.global_forward_projection(x1).detach()
        return p1, z2, p2, z1
    
model = Nine(resnet)
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
model_save_path = 'weight_files/nine+simsiam/0.8G_0.2L_757images_64b_Adam_{epoch}.pth'
save_epochs = []
losses = []
global_losses = []
patch_losses = []
loss_image_folder = './Loss_Image'
epochs = 5000
print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9,global_1,global_2 = batch
        x_vars = [x.permute(0, 3, 1, 2).to(dtype=torch.float).to(device) for x in [x1, x2, x3, x4, x5, x6, x7, x8, x9]]
        y_vars = [y.permute(0, 3, 1, 2).to(dtype=torch.float).to(device) for y in [y1, y2, y3, y4, y5, y6, y7, y8, y9]]
        global_1 = global_1.permute(0, 3, 1, 2).to(dtype=torch.float).to(device)
        global_2 = global_2.permute(0, 3, 1, 2).to(dtype=torch.float).to(device)
        
        p1, z2, p2, z1 = model.simsiam_forward(global_1, global_2)
        global_loss = 0.5*(simsiam_loss(p1, z2) + simsiam_loss(p2, z1))
        global_losses.append(global_loss.detach().item())
        
        zx_vars = [model.forward_prediction(x) for x in x_vars]
        zy_vars = [model.forward_projection(y) for y in y_vars]
        patch_loss = custom_similarity_loss(zx_vars,zy_vars)
        patch_losses.append(patch_loss.detach().item())
        
        loss = 0.8 * global_loss + 0.2 * patch_loss
        total_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if (epoch+1) % save_interval == 0:
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        print(f"Epoch: {epoch+1}, avg loss: {avg_loss:.5f}")
        model_path = model_save_path.format(epoch=epoch+1)
        torch.save(model.state_dict(), model_path)
        print(f"Saving model to {model_path}")
        save_epochs.append(epoch + 1)
        print(f"epoch{epoch+1}的损失", global_loss / batch_count)
        if (epoch + 1) % 1000 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(global_losses, label="Global Loss", color='r')
            plt.plot(patch_losses, label="Patch Loss", color='b')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Loss Curves at Epoch {epoch + 1}")
            plt.legend()
            loss_image_path = os.path.join(loss_image_folder, f"Nine+SimSiam_{epoch + 1}.png")
            plt.savefig(loss_image_path)
            plt.close()





