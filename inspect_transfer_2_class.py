from src.model import NeuralNetwork
import torch


model = NeuralNetwork("efficientnet_b0", num_classes=2, is_pretrained=True, dropout_rate=0)
# model.load_state_dict(torch.load("models/trial_3.pth"))

pretrained_dict = torch.load("models/trial_3.pth")
print(pretrained_dict["base_model._fc.weight"]) 
print(type(pretrained_dict))

for key, value in pretrained_dict.items():
    # print(key)
    # print(value)
    pass
    
print(len(pretrained_dict))

### METHOD 1 ###
# state = model.state_dict()
# state.update(partial)
# model.load_state_dict(state)

### METHOD 2 ###
# def load_my_state_dict(self, state_dict):

#     own_state = self.state_dict()
#     for name, param in state_dict.items():
#         if name not in own_state:
#              continue
#         if isinstance(param, Parameter):
#             # backwards compatibility for serialized parameters
#             param = param.data
#         own_state[name].copy_(param)

### METHOD 3 ###
# pretrained_dict = ...
model_dict = model.state_dict()

# 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# filter unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
print(len(pretrained_dict))
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)

# print(pretrained_dict["base_model._fc.weight"])
print(model_dict["base_model._fc.weight"]) 
# 3. load the new state dict
model.load_state_dict(model_dict)

# print(model)


