"""
Contain the definition of model architecture
"""
import torch, torchvision
import torch.nn as nn
from torchsummary import summary
import resnet_3d
from efficientnet_pytorch_3d import EfficientNet3D
import monai
from torch.autograd import Variable

try:
    from src.architecture.mobilenetv2 import MobileNetV2
    from src.architecture.mobilenet import MobileNet
    from src.architecture.shufflenetv2 import ShuffleNetV2
except:
    from architecture.mobilenetv2 import MobileNetV2
    from architecture.mobilenet import MobileNet
    from architecture.shufflenetv2 import ShuffleNetV2



##### BACKBONE MODELS ######
class NeuralNetwork(nn.Module):
    def __init__(self, model_name, num_classes=4, is_pretrained=True, dropout_rate = 0, image_size = 110):
        super().__init__()
        
        if model_name == "resnet10":
        # Load Resnet50 with pretrained ImageNet weights
            self.base_model = resnet_3d.resnet10_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
        
        elif model_name == "resnet18":
        # Load Resnet50 with pretrained ImageNet weights
            self.base_model = resnet_3d.resnet18_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
        
        elif model_name == "resnet34":
            self.base_model = resnet_3d.resnet34_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
            
        elif model_name == "resnet50":
            self.base_model = resnet_3d.resnet50_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
            
            
        elif model_name == "seresnet50":
            keyword_arguments = {'spatial_dims': 3,
                                'in_channels': 1,
                                "num_classes": num_classes}
            self.base_model = monai.networks.nets.SEResNet50(layers=(3, 4, 6, 3), groups=1, reduction=16, dropout_prob=None, inplanes=64, downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **keyword_arguments)
            classifier_input_size = self.base_model.last_linear.in_features
            self.base_model.last_linear = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True))
            
        elif model_name == "seresnext50":
            keyword_arguments = {'spatial_dims': 3,
                    'in_channels': 1,
                    "num_classes": num_classes}
            self.base_model =  monai.networks.nets.SEResNext50(layers=(3, 4, 6, 3), groups=32, reduction=16, dropout_prob=None, inplanes=64, downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **keyword_arguments)
            classifier_input_size = self.base_model.last_linear.in_features
            self.base_model.last_linear = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True))
                 
        elif model_name == "senet154":
            keyword_arguments = {'spatial_dims': 3,
                    'in_channels': 1,
                    "num_classes": num_classes}
            self.base_model =  monai.networks.nets.SENet154(layers=(3, 8, 36, 3), groups=64, reduction=16, pretrained=False, progress=True, **keyword_arguments)
            self.base_model.dropout = nn.Dropout(p=dropout_rate)
                
            
        elif model_name == "efficientnet_b0":
            self.base_model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes}, in_channels=1)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            # classifier_input_size = self.net.fc.in_features
            # self.net.fc = nn.Sequential(
            #                 nn.Linear(classifier_input_size, num_classes),
            #                 nn.LogSoftMax())
        elif model_name == "efficientnet_b1":
            self.base_model = EfficientNet3D.from_name("efficientnet-b1", override_params={'num_classes': num_classes}, in_channels=1)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "efficientnet_b3":
            self.base_model = EfficientNet3D.from_name("efficientnet-b3", override_params={'num_classes': num_classes}, in_channels=1)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "efficientnet_b7":
            self.base_model = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': num_classes}, in_channels=1)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "efficientnet_b0_bn":
            self.base_model = monai.networks.nets.EfficientNetBN("efficientnet-b0", pretrained=True, progress=True, spatial_dims=3, in_channels=1, num_classes=num_classes, norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "efficientnet_b3_bn":
            self.base_model = monai.networks.nets.EfficientNetBN("efficientnet-b3", pretrained=True, progress=True, spatial_dims=3, in_channels=1, num_classes=num_classes, norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "efficientnet_b7_bn":
            self.base_model = monai.networks.nets.EfficientNetBN("efficientnet-b7", pretrained=True, progress=True, spatial_dims=3, in_channels=1, num_classes=num_classes, norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            
        elif model_name == "densenet121":
            keyword_arguments = {'spatial_dims': 3,
                                'in_channels': 1,
                                'out_channels': num_classes}
            self.base_model = monai.networks.nets.DenseNet121(init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), pretrained=False, progress=True, **keyword_arguments)
            classifier_input_size = self.base_model.class_layers.out.in_features
            self.base_model.class_layers.out = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True))
            
        elif model_name == 'mobilenet':
            self.base_model = MobileNet(num_classes=num_classes, sample_size = image_size, width_mult=1., dropout_rate=dropout_rate)
        
        elif model_name == 'mobilenet_v2':
            self.base_model = MobileNetV2(num_classes=num_classes, sample_size=image_size, width_mult=1., dropout_rate=dropout_rate)
        elif model_name == 'shufflenet_v2':
            self.base_model = ShuffleNetV2(num_classes=num_classes, sample_size=image_size, width_mult=1., dropout_rate=dropout_rate)   
        else:
            raise ValueError("Wrong keywords for model_name argument")
        
    # print(my_model)    
        

    def forward(self, images):
        x = self.base_model(images)
        return x
        # return self.activation(x)
    

if __name__ == "__main__":
    my_model = NeuralNetwork("densenet121", num_classes=4, is_pretrained=True)
    # print(my_model7
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # my_model.to(device)
    # summary(my_model, input_size=(1, 110, 110, 110))
    print(my_model)
    
    input_var = Variable(torch.randn(8, 1, 110, 110, 110))
    print(input_var.shape)
    # input_var.to(device)
    # print(type(input_var))
    output = my_model(input_var)
    print(output.shape)