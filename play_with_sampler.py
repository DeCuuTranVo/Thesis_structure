import torch
from src.dataset import CustomImageDataset

X_TRAIN_PATH = "tensor/x_train_tensor.pt"
X_TEST_PATH = "tensor/x_test_tensor.pt"
Y_TRAIN_PATH = "tensor/y_train_tensor.pt"
Y_TEST_PATH = "tensor/y_test_tensor.pt"



x_train = torch.load(X_TRAIN_PATH)
x_test = torch.load(X_TEST_PATH)
y_train = torch.load(Y_TRAIN_PATH)
y_test = torch.load(Y_TEST_PATH)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = torch.unsqueeze(x_train, 1)
x_test = torch.unsqueeze(x_test, 1)
y_train = y_train.type(torch.long) #long
y_test = y_test.type(torch.long) #long


train_dataset = CustomImageDataset(
    x_train,
    y_train)  
test_dataset = CustomImageDataset(
    x_test,
    y_test)

print(train_dataset)
print(test_dataset)

distinct_label_train, counts_label_train =  torch.unique(y_train, return_counts=True)
distinct_label_test, counts_label_test = torch.unique(y_test, return_counts=True)

print(distinct_label_train, counts_label_train)
print(distinct_label_test, counts_label_test)

num_label_total = len(y_train) + len(y_test)
print(num_label_total)

class_weight = []
for label in distinct_label_train.tolist():
    class_weight.append(1.0/(counts_label_train[label]+counts_label_test[label])) #float(num_label_total)
        
class_weight = torch.FloatTensor(class_weight)

print(class_weight)

index_sequence = []
weight_sequence = []

for i in range(len(y_train)):
    for item in distinct_label_train.tolist():
        if y_train[i] == item:
            index_sequence.append(i)
            weight_sequence.append(class_weight[item])
     
print("index_sequence")   
print(len(index_sequence))   
print(index_sequence)
   
print("weight_sequence") 
print(len(weight_sequence))      
print(weight_sequence)



