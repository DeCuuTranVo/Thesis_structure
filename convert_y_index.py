import torch

y_train_NC_AD = torch.load("tensor/y_train_NC_AD.pt")
y_test_NC_AD = torch.load("tensor/y_test_NC_AD.pt")

for i in range(y_train_NC_AD.shape[0]):
    if (y_train_NC_AD[i] == 3):
        y_train_NC_AD[i] = 1
        
for i in range(y_test_NC_AD.shape[0]):
    if (y_test_NC_AD[i] == 3):
        y_test_NC_AD[i] = 1
        
torch.save(y_train_NC_AD, "tensor/y_train_NC_AD.pt")
torch.save(y_test_NC_AD, "tensor/y_test_NC_AD.pt")