import torch.optim as optim
import torch.nn as nn
import resnet_3d
from torchsummary import summary

TARGET_WIDTH=110
TARGET_HEIGHT=110
TARGET_DEPTH=110
model = resnet_3d.resnet18_3d(pretrained = True, input_channels = 1, num_classes=4)
device = "cuda:0"
model.to(device)
summary(model, input_size=(1, TARGET_WIDTH, TARGET_HEIGHT, TARGET_DEPTH))
######################################################################################
# Import M3d-CAM
from medcam import medcam
# Inject model with M3d-CAM
model = medcam.inject(model, output_dir="attention_maps", save_maps=True, layer="auto")
print(medcam.get_layers(model))

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 64, 55, 55, 55]          21,952
       BatchNorm3d-2       [-1, 64, 55, 55, 55]             128
              ReLU-3       [-1, 64, 55, 55, 55]               0
         MaxPool3d-4       [-1, 64, 28, 28, 28]               0
            Conv3d-5       [-1, 64, 28, 28, 28]         110,592
       BatchNorm3d-6       [-1, 64, 28, 28, 28]             128
              ReLU-7       [-1, 64, 28, 28, 28]               0
            Conv3d-8       [-1, 64, 28, 28, 28]         110,592
       BatchNorm3d-9       [-1, 64, 28, 28, 28]             128
             ReLU-10       [-1, 64, 28, 28, 28]               0
       BasicBlock-11       [-1, 64, 28, 28, 28]               0
           Conv3d-12       [-1, 64, 28, 28, 28]         110,592
      BatchNorm3d-13       [-1, 64, 28, 28, 28]             128
             ReLU-14       [-1, 64, 28, 28, 28]               0
           Conv3d-15       [-1, 64, 28, 28, 28]         110,592
      BatchNorm3d-16       [-1, 64, 28, 28, 28]             128
             ReLU-17       [-1, 64, 28, 28, 28]               0
       BasicBlock-18       [-1, 64, 28, 28, 28]               0
           Conv3d-19      [-1, 128, 14, 14, 14]         221,184
      BatchNorm3d-20      [-1, 128, 14, 14, 14]             256
             ReLU-21      [-1, 128, 14, 14, 14]               0
           Conv3d-22      [-1, 128, 14, 14, 14]         442,368
      BatchNorm3d-23      [-1, 128, 14, 14, 14]             256
           Conv3d-24      [-1, 128, 14, 14, 14]           8,192
      BatchNorm3d-25      [-1, 128, 14, 14, 14]             256
             ReLU-26      [-1, 128, 14, 14, 14]               0
       BasicBlock-27      [-1, 128, 14, 14, 14]               0
           Conv3d-28      [-1, 128, 14, 14, 14]         442,368
      BatchNorm3d-29      [-1, 128, 14, 14, 14]             256
             ReLU-30      [-1, 128, 14, 14, 14]               0
           Conv3d-31      [-1, 128, 14, 14, 14]         442,368
      BatchNorm3d-32      [-1, 128, 14, 14, 14]             256
             ReLU-33      [-1, 128, 14, 14, 14]               0
       BasicBlock-34      [-1, 128, 14, 14, 14]               0
           Conv3d-35         [-1, 256, 7, 7, 7]         884,736
      BatchNorm3d-36         [-1, 256, 7, 7, 7]             512
             ReLU-37         [-1, 256, 7, 7, 7]               0
           Conv3d-38         [-1, 256, 7, 7, 7]       1,769,472
      BatchNorm3d-39         [-1, 256, 7, 7, 7]             512
           Conv3d-40         [-1, 256, 7, 7, 7]          32,768
      BatchNorm3d-41         [-1, 256, 7, 7, 7]             512
             ReLU-42         [-1, 256, 7, 7, 7]               0
       BasicBlock-43         [-1, 256, 7, 7, 7]               0
           Conv3d-44         [-1, 256, 7, 7, 7]       1,769,472
      BatchNorm3d-45         [-1, 256, 7, 7, 7]             512
             ReLU-46         [-1, 256, 7, 7, 7]               0
           Conv3d-47         [-1, 256, 7, 7, 7]       1,769,472
      BatchNorm3d-48         [-1, 256, 7, 7, 7]             512
             ReLU-49         [-1, 256, 7, 7, 7]               0
       BasicBlock-50         [-1, 256, 7, 7, 7]               0
           Conv3d-51         [-1, 512, 4, 4, 4]       3,538,944
      BatchNorm3d-52         [-1, 512, 4, 4, 4]           1,024
             ReLU-53         [-1, 512, 4, 4, 4]               0
           Conv3d-54         [-1, 512, 4, 4, 4]       7,077,888
      BatchNorm3d-55         [-1, 512, 4, 4, 4]           1,024
           Conv3d-56         [-1, 512, 4, 4, 4]         131,072
      BatchNorm3d-57         [-1, 512, 4, 4, 4]           1,024
             ReLU-58         [-1, 512, 4, 4, 4]               0
       BasicBlock-59         [-1, 512, 4, 4, 4]               0
           Conv3d-60         [-1, 512, 4, 4, 4]       7,077,888
      BatchNorm3d-61         [-1, 512, 4, 4, 4]           1,024
             ReLU-62         [-1, 512, 4, 4, 4]               0
           Conv3d-63         [-1, 512, 4, 4, 4]       7,077,888
      BatchNorm3d-64         [-1, 512, 4, 4, 4]           1,024
             ReLU-65         [-1, 512, 4, 4, 4]               0
       BasicBlock-66         [-1, 512, 4, 4, 4]               0
AdaptiveAvgPool3d-67         [-1, 512, 1, 1, 1]               0
           Linear-68                    [-1, 4]           2,052
================================================================
Total params: 33,162,052
Trainable params: 33,162,052
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 5.08
Forward/backward pass size (MB): 462.09
Params size (MB): 126.50
Estimated Total Size (MB): 593.67
----------------------------------------------------------------
['conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 
'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 
'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1', 'layer1', 'layer2.0.conv1', 'layer2.0.bn1', 
'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 
'layer2.0.downsample', 'layer2.0', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu',
'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1', 'layer2', 'layer3.0.conv1', 'layer3.0.bn1', 
'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 
'layer3.0.downsample', 'layer3.0', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2',
'layer3.1.bn2', 'layer3.1', 'layer3', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2',
'layer4.0.bn2', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.downsample', 'layer4.0', 
'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1', 
'layer4', 'avgpool', 'fc']
"""