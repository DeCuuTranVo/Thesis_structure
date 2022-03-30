import torch
import sys

X_TRAIN_PATH = "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/x_train_NC_EMCI_AD.pt"
X_TEST_PATH = "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/x_test_NC_EMCI_AD.pt"
Y_TRAIN_PATH = "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_train_NC_EMCI_AD.pt"
Y_TEST_PATH = "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_test_NC_EMCI_AD.pt"

x_train = torch.load(X_TRAIN_PATH)
x_test = torch.load(X_TEST_PATH)
y_train = torch.load(Y_TRAIN_PATH)
y_test = torch.load(Y_TEST_PATH)

# print(x_train.shape, y_train.shape, y_test.shape, y_test.shape)

x = [x_train, x_test]
y = [y_train, y_test]

x = torch.cat(x, dim=0)
y = torch.cat(y, dim=0)

torch.save(x, "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/x_tensor_NC_EMCI_AD_cv.pt")
torch.save(y, "/mnt/data_lab513/vqtran_data/data_torch_tensor/tensor/y_tensor_NC_EMCI_AD_cv.pt")
print(x.shape)
print(y.shape)


