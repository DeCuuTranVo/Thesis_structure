"""
Contain the definition of model architecture
"""
import torch, torchvision
import torch.nn as nn
from torchsummary import summary
import resnet_3d
from efficientnet_pytorch_3d import EfficientNet3D



##### BACKBONE MODELS ######
class NeuralNetwork(nn.Module):
    def __init__(self, model_name, num_classes=4, is_pretrained=True, dropout_rate = 0):
        super().__init__()
        
        if model_name == "resnet18":
        # Load Resnet50 with pretrained ImageNet weights
            self.base_model = resnet_3d.resnet18_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes))
            #                 nn.LogSoftMax())
            
        if model_name == "efficientnet_b0":
            self.base_model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes}, in_channels=1)
            self.base_model._dropout = nn.Dropout(p = dropout_rate)
            # classifier_input_size = self.net.fc.in_features
            # self.net.fc = nn.Sequential(
            #                 nn.Linear(classifier_input_size, num_classes),
            #                 nn.LogSoftMax())
            
        # self.activation = nn.LogSoftmax()
        # self.activation = torch.nn.LogSoftmax(dim=1)

    def forward(self, images):
        x = self.base_model(images)
        return x
        # return self.activation(x)
    

if __name__ == "__main__":
    my_model = NeuralNetwork("efficientnet_b0", num_classes=4, is_pretrained=True)
    print(my_model)
    # summary(my_model, input_size=(1, 110, 110, 110))