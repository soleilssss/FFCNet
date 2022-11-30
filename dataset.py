'''
加载数据的原图，局部的图形。
'''
import PIL.Image as Image 
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def make_dataset(path):
    imgs = []
    img_types = ['adenoma','cancer','normal','polyp']
    for i in range(4):
        img_dir_path = os.path.join(path,img_types[i],'image')
        img_name = os.listdir(img_dir_path)
        for name in img_name:
            img_path = os.path.join(img_dir_path,name)
            imgs.append((img_path, i))
    return imgs

def swap(img, crop, p):
    def crop_image(image, cropnum):
        width, high = image.shape[1], image.shape[2]
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                img = image[:, crop_y[j]:min(crop_y[j + 1], high), crop_x[i]:min(crop_x[i + 1], width)]
                img = np.fft.fftshift(np.fft.fftn(img))
                # img = np.abs(img)
                im_list.append(img)
        return im_list

    images = crop_image(img, crop)
    if p > random.random():
        random.shuffle(images)
    width, high = img.shape[0],img.shape[1]
    iw = int(width / crop[0])
    ih = int(high / crop[1])
    img1 = []
    for i in range(crop[0]):
        img1.append(np.concatenate(images[i*crop[0]:(i+1)*crop[0]],1))
    toImage = np.concatenate(img1,2)

    return toImage


class BowelDataset(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        crop_path, label = self.imgs[index]
        img = Image.open(crop_path)
        if self.transform is not None:
            img = self.transform(img)
            img = np.array(img)
            img = img.astype(np.float32)
            img = img/255.0
            img = img.transpose((2, 0, 1))
            img_F = swap(img, (40, 40), p=0.1)
            F_real = np.real(img_F)
            F_imag = np.imag(img_F)
            F_complex = np.concatenate((F_real, F_imag), axis=0)
            F_complex = torch.tensor(F_complex).float()
            label = torch.tensor(label)
        return F_complex, label
            

    def __len__(self):
        return len(self.imgs)

    def getPath(self):
        return self.imgs


