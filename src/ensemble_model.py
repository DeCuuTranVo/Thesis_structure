import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torch.utils.tensorboard import SummaryWriter



try:
    from src.model import NeuralNetwork 
except:
    from model import NeuralNetwork 

class ResNetEfficientNetEnsemble(nn.Module):
    def __init__(self, model_resnet, model_efficientnet, nb_classes=10):
        super(ResNetEfficientNetEnsemble, self).__init__()
        self.model_resnet = model_resnet
        self.model_efficientnet = model_efficientnet
        # Remove last linear layer
        model_resnet_in_value = self.model_resnet.base_model.fc[1].in_features
        model_efficientnet_in_value = self.model_efficientnet.base_model._fc.in_features
        # print(model_resnet_in_value)
        # print(model_efficientnet_in_value)
        
        self.model_resnet.base_model.fc[1] = nn.Identity()
        self.model_efficientnet.base_model._fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(model_resnet_in_value + model_efficientnet_in_value, nb_classes)
        
    def forward(self, x):
        x1 = self.model_resnet(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_efficientnet(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(F.relu(x))
        return x
    
class ResNetEfficientNetShuffleNetEnsemble(nn.Module):
    def __init__(self, model_resnet, model_efficientnet, model_shufflenet, nb_classes=10):
        super(ResNetEfficientNetShuffleNetEnsemble, self).__init__()
        self.model_resnet = model_resnet
        self.model_efficientnet = model_efficientnet
        self.model_shufflenet= model_shufflenet
        
        
        model_resnet_in_value = self.model_resnet.base_model.fc[1].in_features #Uncomment this line
        model_efficientnet_in_value = self.model_efficientnet.base_model._fc.in_features #Uncomment this line
        model_shufflenet_in_value = self.model_shufflenet.base_model.classifier[1].in_features #Uncomment this line
        
        # model_resnet_out_value = self.model_resnet.base_model.fc[1].out_features
        # model_efficientnet_out_value = self.model_efficientnet.base_model._fc.out_features
        # model_shufflenet_out_value = self.model_shufflenet.base_model.classifier[1].out_features
        
        # print(modelA_in_value)
        # print(modelB_in_value)
        # print(modelC_in_value)
        
        # Remove last linear layer
        self.model_resnet.base_model.fc[1] = nn.Identity() #Uncomment this line
        self.model_efficientnet.base_model._fc = nn.Identity() #Uncomment this line
        self.model_shufflenet.base_model.classifier[1] = nn.Identity() #Uncomment this line
        
        # Create new classifier
        self.classifier = nn.Linear(model_resnet_in_value + model_efficientnet_in_value + model_shufflenet_in_value, nb_classes) #Uncomment this line
        # self.classifier = nn.Linear(model_resnet_out_value + model_efficientnet_out_value + model_shufflenet_out_value, nb_classes)
        
    def forward(self, x):
        x1 = self.model_resnet(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_efficientnet(x.clone())
        x2 = x2.view(x2.size(0), -1)
        x3 = self.model_shufflenet(x)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

class ResNetDenseNetShuffleNetEnsemble(nn.Module):
    def __init__(self, model_resnet, model_densenet, model_shufflenet, nb_classes=10):
        super(ResNetDenseNetShuffleNetEnsemble, self).__init__()
        self.model_resnet = model_resnet
        self.model_densenet = model_densenet
        self.model_shufflenet= model_shufflenet
        
        
        # model_resnet_in_value = self.model_resnet.base_model.fc[1].in_features #Uncomment this line
        # model_densenet_in_value = self.model_densenet.base_model.class_layers.out[1].in_features #Uncomment this line
        # model_shufflenet_in_value = self.model_shufflenet.base_model.classifier[1].in_features #Uncomment this line
        
        model_resnet_out_value = self.model_resnet.base_model.fc[1].out_features
        model_densenet_out_value = self.model_densenet.base_model.class_layers.out[1].out_features
        model_shufflenet_out_value = self.model_shufflenet.base_model.classifier[1].out_features
        
        # # Remove last linear layer
        # self.model_resnet.base_model.fc[1] = nn.Identity() #Uncomment this line
        # self.model_densenet.base_model.class_layers[1] = nn.Identity() #Uncomment this line
        # self.model_shufflenet.base_model.classifier[1] = nn.Identity() #Uncomment this line
        
        # Create new classifier
        # self.classifier = nn.Linear(model_resnet_in_value + model_densenet_in_value + model_shufflenet_in_value, nb_classes) #Uncomment this line
        self.classifier = nn.Linear(model_resnet_out_value + model_densenet_out_value + model_shufflenet_out_value, nb_classes)
        
    def forward(self, x):
        x1 = self.model_resnet(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_densenet(x.clone())
        x2 = x2.view(x2.size(0), -1)
        x3 = self.model_shufflenet(x)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.classifier(F.relu(x))
        return x
    
class ThreeModelEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, nb_classes=10):
        super(ThreeModelEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        
        
        modelA_in_value = self.modelA.base_model.fc.in_features
        modelB_in_value = self.modelB.base_model.fc.in_features
        modelC_in_value = self.modelC.base_model.fc.in_features
        
        # print(modelA_in_value)
        # print(modelB_in_value)
        # print(modelC_in_value)
        
        # Remove last linear layer
        self.modelA.base_model.fc = nn.Identity()
        self.modelB.base_model.fc = nn.Identity()
        self.modelC.base_model.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(modelA_in_value + modelB_in_value + modelC_in_value, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x.clone())
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelC(x)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.classifier(F.relu(x))
        return x


if __name__ == '__main__':
    model_resnet = NeuralNetwork("resnet18", num_classes=2, is_pretrained=True, dropout_rate = 0, image_size = 110)
    model_efficientnet = NeuralNetwork("efficientnet_b0", num_classes=2, is_pretrained=True, dropout_rate = 0, image_size = 110)
    model_shufflenet = NeuralNetwork("shufflenet_v2", num_classes=2, is_pretrained=True, dropout_rate = 0, image_size = 110)
    model_densenet = NeuralNetwork("densenet121", num_classes=2, is_pretrained=True, dropout_rate = 0, image_size = 110)
    # UnFreeze these models
    for param in model_resnet.parameters():
        param.requires_grad_(True)

    for param in model_efficientnet.parameters():
        param.requires_grad_(True)
        
    for param in model_shufflenet.parameters():
        param.requires_grad_(True)

    #Create ensemble model from 2 element model
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/model_resnet_densenet_shufflenet')
    model_resnet_densenet_shufflenet = ResNetDenseNetShuffleNetEnsemble(model_resnet, model_densenet, model_shufflenet, nb_classes=2)
    x =  Variable(torch.randn(8, 1, 110, 110, 110))
    writer.add_graph(model_resnet_densenet_shufflenet, x)
    print(x.shape)
    output = model_resnet_densenet_shufflenet(x)
    print(model_resnet_densenet_shufflenet)
    print(output.shape)



    # Create ensemble model from 3 element model
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/model_resnet_efficientnet_shufflenet/trial_161')
    # model_resnet_efficientnet_shufflenet = ResNetEfficientNetShuffleNetEnsemble(model_resnet, model_efficientnet, model_shufflenet, nb_classes=2)
    # x =  Variable(torch.randn(8, 1, 110, 110, 110))
    # writer.add_graph(model_resnet_efficientnet_shufflenet, x)
    # print(x.shape)
    # output = model_resnet_efficientnet_shufflenet(x)
    # print(model_resnet_efficientnet_shufflenet)
    # print(output.shape)


    # # Create ensemble model from 3 element model
    # model = ThreeModelEnsemble(modelA, modelB, modelC)
    # x = torch.randn(1,1,110, 110, 110)
    # output = model(x)
    # print(output.shape)
    # print(model)

    writer.close()