import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import pandas as pd
import json
from src.utils import preprocess
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Generator(Dataset):    
    def __init__(self, df, data_dir, transform=None, target_transform=None):
        image_size = [224, 224]
        image_transformation = [
            transforms.Resize(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        image_transformation = transforms.Compose(image_transformation)

        self.img_labels = df
        self.img_dir = data_dir
        self.transform = image_transformation
        self.target_transform = target_transform

        self.image_list = []
        self.label_list = []
        # self.image_array = torch.Tensor(3,224,224)
        # self.label_array = torch.Tensor(6)

        # print(self.img_labels)
        print("generator constructor success!")

    def __len__(self):
        return len(self.img_labels)

    def __call__(self,  train=False): #, train=False
        # print("idx: " + str(idx))
        # print(self.img_labels.iloc[idx, 0])
        # print(self.img_labels.iloc[idx, 1])
        # print()
        # print(len(self.img_labels))
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]))
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)    

            label = (torch.FloatTensor(label))
            # print(type(image))
            # print(image.shape)

            # print(type(label))
            # print(type(label[0]))
            
            # label = torch.
            # print(image)
            # print(label)
            # self.image_array = torch.cat((torch.unsqueeze(self.image_array, dim=0), torch.unsqueeze(image, dim =0)), dim=0)
            # self.label_array = torch.cat((self.label_array, label), dim=0) 

            self.image_list.append(image)
            self.label_list.append(label)

        # self.image_array = torch.Tensor(len(self.image_list), self.image_list[0].shape[0], self.image_list[0].shape[1], self.image_list[0].shape[2])
        image_array = torch.stack(self.image_list, dim=0)
        label_array = torch.stack(self.label_list, dim=0)
        print(image_array.shape)
        # self.image_array = torch.cat((self.image_list, ).shape)
        print(label_array.shape)

        # print(len(self.image_array))
        # print(len(self.label_array))

        # print(self.image_array[0])
        # print(self.label_array[0])

        # print(self.image_array.shape)
        # print(self.label_array.shape)

        if train:
            torch.save(image_array, "tensor/train_imgs_array.pt")
            torch.save(label_array, "tensor/train_labels_array.pt")
        else:
            torch.save(image_array, "tensor/test_imgs_array.pt")
            torch.save(label_array, "tensor/test_labels_array.pt")
        print(".pt file saved!")

        # label = (torch.FloatTensor(label))

        # # print(label)
        # # print(label.dtype)
        # print((label.long()).dtype)
        return image_array, label_array

import timeit
if __name__ == "__main__":
    DATA_DIR = "../data/data"
    CSV_DIR = "../pseudolabel_done.csv"
    # Get train and test dataframe from data directory
    df_train, df_test = preprocess(DATA_DIR, CSV_DIR)

    start = timeit.default_timer()
    my_augmentator = Generator(df = df_train , data_dir = DATA_DIR)
    transformed_img, transformed_label =  my_augmentator()
    #Your statements here
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    torch.save(transformed_img, "transformed_img.pt")
    torch.save(transformed_label, "transformed_label.pt")




 