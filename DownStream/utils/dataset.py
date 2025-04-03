import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import copy

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name, data_mean, data_std):
        # 初始化函数，读取所有data_path下的图片
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.data_mean = data_mean
        self.data_std = data_std

        if self.dataset_name == "gendata":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.png'))

        if self.dataset_name == "drive" or self.dataset_name == "chase":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "stare":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "rim":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.png'))
        if self.dataset_name == "gs":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.png'))
        if self.dataset_name == "hei-med":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "idrid":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        self.is_train = is_train

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        if self.dataset_name == "drive":
            label_path = image_path.replace('img', 'label')
            if self.is_train == 1:
                label_path = label_path.replace('_training.tif', '_manual1.tif') 
            else:
                label_path = label_path.replace('_test.tif', '_manual1.tif') 

        if self.dataset_name == "chase":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.tif', '_1stHO.tif') 

        if self.dataset_name == "gendata":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.jpg', '.png') 

        if self.dataset_name == "stare":
            label_path = image_path.replace('image', 'label')

        if self.dataset_name == "rim":
            label_path = image_path.replace('image', 'label')
        if self.dataset_name == "gs":
            label_path = image_path.replace('image', 'label')
        if self.dataset_name == "hei-med":
            label_path = image_path.replace('image', 'label')
            label_path = label_path.replace('.jpg', '.tif') 
        if self.dataset_name == "idrid":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.jpg', '.tif') 
        image = Image.open(image_path)
        label = Image.open(label_path)

        label = label.convert('L')
        raw_height = image.size[1]
        raw_width = image.size[0]
        if self.dataset_name == "rim":
            image = image.resize((256, 256))
            label = label.resize((256, 256))
        if self.dataset_name == "idrid":
            image = image.resize((512, 512))
            label = label.resize((512, 512))
        if self.dataset_name == "gs":
            image = image.resize((256, 256))
            label = label.resize((256, 256))
        if self.dataset_name == "hei-med":
            image = image.resize((900, 900))
            label = label.resize((900, 900))
        if self.dataset_name == "drive":
            image, label = self.padding_image(image, label, 608, 608)
        if self.dataset_name == "stare":
            image, label = self.padding_image(image, label, 706, 706)
        if self.dataset_name == "chase":
            image, label = self.padding_image(image, label, 1024, 1024)
        # Online augmentation
        if self.is_train == 1:
            if torch.rand(1).item() <= 0.5:
                image, label = self.randomRotation(image, label)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

            if self.dataset_name == "gendata":
                color_jitter = transforms.ColorJitter(brightness=0.5,contrast=0.5, saturation=0.5, hue=0.5)
                image = color_jitter(image)
        
        image = np.asarray(image)
        label = np.asarray(label)
        label = label.copy()
        #if label.max() > 1:
            #label = label / 255
        if label.max() > 1:
            label[label>1] = 1
        label = label.reshape(1, label.shape[0], label.shape[1])
        label = np.array(label)

        # whiten
        image = image / 255
        #image = image - np.array(self.data_mean)
        #image = image / np.array(self.data_std)
        image = image.transpose(2, 0, 1)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[4:len(filename)-4] # del .tif
        # stare filename = filename[6:len(filename) - 4]  # del .tif
        # drive filename = filename[4:len(filename)-4] # del .tif
        # filename 用来命名预测的图像，
        return image, label, filename, raw_height, raw_width

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = torch.randint(low=0,high=360,size=(1,1)).long().item()
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        #新建长宽608像素，背景色为（0, 0, 0）的画布对象,即背景为黑，RGB是彩色三通道图像，P是8位图像
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        # 把新建的图像粘贴在原图上
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label
    def cutout(self, img):
        n_holes = 49
        length = 21
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h,w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
