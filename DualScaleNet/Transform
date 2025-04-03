import cv2
import numpy as np
import random
from torchvision import transforms
 
class AugmentedImageSplitter:
    def __init__(self,augmentations=None):
        self.augmentations = augmentations or [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1)
        ]
    def split_and_augment(self, image):
        image = cv2.resize(image, (224, 224))
        h,w = image.shape[:2]
        patch_h,patch_w = h // 3,w//3
        images = [image.copy(),image.copy()]
        patches = []
        for img in images:
            for i in range(3):
                for j in range(3):
                    patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    patch_tensor = transforms.ToTensor()(patch)
                    aug_transform = random.choice(self.augmentations)
                    augmented_patch = aug_transform(patch_tensor)
                    augmented_patch_np = transforms.ToPILImage()(augmented_patch).convert("RGB")
                    patches.append(np.array(augmented_patch_np))
        global_views = []
        for _ in range(2):  # Generate two global views
            image_tensor = transforms.ToTensor()(image)
            aug_transform = random.choice(self.augmentations)
            augmented_image = aug_transform(image_tensor)
            augmented_image_np = transforms.ToPILImage()(augmented_image).convert("RGB")
            global_views.append(np.array(augmented_image_np))
        return patches + global_views
