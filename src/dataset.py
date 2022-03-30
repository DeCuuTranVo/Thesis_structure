import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import pandas as pd
import torchio as tio
import monai


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = [224, 224]


class CustomImageDataset(Dataset):
    def __init__(
            self,
            img_tensor,
            label_tensor,
            transform=None,
            target_transform=None,
            augment=True,
            train_or_val = "train",
            rotation=0,
            translation=None,
            scaling=None):
        """Construct dataset object, prepare image for dataloader

        Args:
            img_tensor (torch.Tensor): Image tensor of shape [batch_size, 3, 224, 224]
            label_tensor (torch.Tensor): Label tensor of shape [batch_size, 6]
            transform (torchvision.transforms.transforms.Compose, optional): The combination of transformations applied on image. Defaults to None.
            target_transform (Callable, optional): The transformations applied on label tensor. Defaults to None.
            augment (bool, optional): Whether to apply the augmentation transformation or not. Defaults to True.
            rotation (int or list, optional): range of rotation degree of augmentation. Defaults to 0.
            translation (tupple or list, optional): range of translation degree of augmentation. Defaults to None.
            scaling (tupple or list, optional): range of scaling degree of augmentation. Defaults to None.
        """
        # transforms_dict = {
        #     tio.
        #     tio.RandomElasticDeformation(): 0.25,
        # }  # Using 3 and 1 as probabilities would have the same effect

        # print("init dataset")        
        image_transformation_tio= (
            # tio.RandomAffine(scales=scaling, degrees=rotation, translation=translation, center="image"),
            tio.RandomAffine(scales=(0.8,1.2), degrees=(-20,20), translation=(0,0.2), center="image"), #????
            tio.RandomFlip(axes=['LR']),
            # tio.RandomElasticDeformation(
            #     num_control_points=(7, 7, 7),  # or just 7
            #     locked_borders=2,
            # ),
            # tio.RandomNoise(),
            # tio.RandomGamma(),

            # # tio.OneOf(transforms_dict),
            # tio.RandomBlur(),
            # tio.ZNormalization(),
            # tio.RescaleIntensity(),
        )
        
        image_transformation_tio_test = (
            # 
            # tio.transforms.ZNormalization(),
            # tio.RescaleIntensity(),
        )
        
        image_transformation_monai = (            
            # monai.transforms.HistogramNormalize(),
            # monai.transforms.NormalizeIntensity(),
            # monai.transforms.RandStdShiftIntensity(0.9, prob=0.1),
            # monai.transforms.RandRotate90(prob=0.1, max_k=3, spatial_axes=(0, 1)),
            # monai.transforms.RandAffine(prob=0.9, rotate_range=(-0.087, 0.087), shear_range=None, translate_range=(0,0.1), scale_range=(0.8,1.2)), 
            # monai.transforms.RandFlip(prob=0.5, spatial_axis=None),
            # monai.transforms.ToTensor(dtype=None, device=None)
            #monai.transforms.CastToType(dtype=<class 'numpy.float32'>)
        )
        
        image_transformation_tio_train = tio.Compose(image_transformation_tio)
        image_transformation_tio_test = tio.Compose(image_transformation_tio_test)
        image_transformation_monai = monai.transforms.Compose(image_transformation_monai)

        self.img_tensor = img_tensor
        self.label_tensor = label_tensor

        if transform is None:
            self.transform = None
            if train_or_val == "train":
                self.transform_1 = image_transformation_tio_train
                # print("transformation of train set")
            elif train_or_val == "val":
                self.transform_1 = image_transformation_tio_test
                # print("transformation of test set")
            else:
                raise ValueError("Wrong arguments, only 2 options: train or val")
            
            self.transform_2 = image_transformation_monai
        else:
            self.transform = transform

        self.target_transform = target_transform
        self.augment = augment
        
        # print("use augmentation", self.augment)
        # print("augmentation type", self.transform_1, self.transform_2)

    def __len__(self):
        """ Get the number of samples in the dataset
        """
        return len(self.img_tensor)

    def __getitem__(self, idx):
        """ Get the image and label of index "idx" in the dataset

        Args:
            idx (int): index number of a sample in the dataset

        Returns:
            image (torch.Tensor): image tensor of a sample with shape [3, 224, 224]
            label (torch.Tensor): label tensor of a sample with shape [6]
        """
        image = self.img_tensor[idx]
        label = self.label_tensor[idx]
        image = torch.squeeze(image, 0) # uncommnet this line for the training of unfiltered problems
        if self.augment:
            if self.transform is None:
                # image = torch.unsqueeze(image, 0)
                
                image = self.transform_1(image)
                # image = self.transform_2(image)
                
                # image = torch.squeeze(image)
            else:

                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_labels(self):
        return self.label_tensor