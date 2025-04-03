import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Fundus_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image)
        if self.transform:
            all_elements = self.transform.split_and_augment(image)
            normalized_elements = [patch / 255.0 for patch in all_elements]
            return normalized_elements
