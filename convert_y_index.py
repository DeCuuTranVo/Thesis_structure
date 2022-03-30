import torch

y_train_NC_EMCI_AD = torch.load("/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_train_NC_EMCI_AD.pt")
y_test_NC_EMCI_AD = torch.load("/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_test_NC_EMCI_AD.pt")

for i in range(y_train_NC_EMCI_AD.shape[0]):
    if (y_train_NC_EMCI_AD[i] == 3):
        y_train_NC_EMCI_AD[i] = 2
        
for i in range(y_test_NC_EMCI_AD.shape[0]):
    if (y_test_NC_EMCI_AD[i] == 3):
        y_test_NC_EMCI_AD[i] = 2
        
torch.save(y_train_NC_EMCI_AD, "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_train_NC_EMCI_AD.pt")
torch.save(y_test_NC_EMCI_AD, "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_test_NC_EMCI_AD.pt")