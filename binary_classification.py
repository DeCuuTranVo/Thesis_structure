import torch

x_train = torch.load("tensor/x_train_tensor.pt")
x_test = torch.load("tensor/x_test_tensor.pt")
y_train = torch.load("tensor/y_train_tensor.pt")
y_test = torch.load("tensor/y_test_tensor.pt")

#####################################INSPECT DATASET#####################################3
#CN:0, EMCI:1, LMCI:2, AD:3
print(y_train.shape)
print(y_test.shape)


# num_CN = 0
# num_EMCI = 0
# num_LMCI = 0
# num_AD = 0 
class_distribution_train = [0,0,0,0]
class_distribution_test = [0,0,0,0]

for i in range(y_train.shape[0]):
    for j in range(len(class_distribution_train)):
        if y_train[i] == j:
            class_distribution_train[j]+=1

for i in range(y_test.shape[0]):
    for j in range(len(class_distribution_test)):
        if y_test[i] == j:
            class_distribution_test[j]+=1    
print(class_distribution_train) #[279, 191, 111, 223]
print(class_distribution_test) #[70, 47, 28, 56]


##################################AD vs NC dataset#######################################
x_train_NC_AD = []
x_test_NC_AD = []
y_train_NC_AD = []
y_test_NC_AD = []

for i in range(y_train.shape[0]):
    if (y_train[i] == 0) or (y_train[i] == 3):
        x_train_NC_AD.append(x_train[i])
        y_train_NC_AD.append(y_train[i])
        
for i in range(y_test.shape[0]):
    if (y_test[i] == 0) or (y_test[i] == 3):
        x_test_NC_AD.append(x_test[i])
        y_test_NC_AD.append(y_test[i])

print(len(x_train_NC_AD))
print(len(x_test_NC_AD))
print(len(y_train_NC_AD))
print(len(y_test_NC_AD))

x_train_NC_AD = torch.stack(x_train_NC_AD)
x_test_NC_AD = torch.stack(x_test_NC_AD)
y_train_NC_AD = torch.stack(y_train_NC_AD)
y_test_NC_AD = torch.stack(y_test_NC_AD)

print(x_train_NC_AD.shape)
print(x_test_NC_AD.shape)
print(y_train_NC_AD.shape)
print(y_test_NC_AD.shape)

torch.save(x_train_NC_AD, "x_train_NC_AD.pt")
torch.save(x_test_NC_AD, "x_test_NC_AD.pt")
torch.save(y_train_NC_AD, "y_train_NC_AD.pt")
torch.save(y_test_NC_AD, "y_test_NC_AD.pt")

##################################NC vs EMCI dataset#######################################
x_train_NC_EMCI = []
x_test_NC_EMCI = []
y_train_NC_EMCI = []
y_test_NC_EMCI = []

for i in range(y_train.shape[0]):
    if (y_train[i] == 0) or (y_train[i] == 1):
        x_train_NC_EMCI.append(x_train[i])
        y_train_NC_EMCI.append(y_train[i])
        
for i in range(y_test.shape[0]):
    if (y_test[i] == 0) or (y_test[i] == 1):
        x_test_NC_EMCI.append(x_test[i])
        y_test_NC_EMCI.append(y_test[i])

print(len(x_train_NC_EMCI))
print(len(x_test_NC_EMCI))
print(len(y_train_NC_EMCI))
print(len(y_test_NC_EMCI))

x_train_NC_EMCI = torch.stack(x_train_NC_EMCI)
x_test_NC_EMCI = torch.stack(x_test_NC_EMCI)
y_train_NC_EMCI = torch.stack(y_train_NC_EMCI)
y_test_NC_EMCI = torch.stack(y_test_NC_EMCI)

print(x_train_NC_EMCI.shape)
print(x_test_NC_EMCI.shape)
print(y_train_NC_EMCI.shape)
print(y_test_NC_EMCI.shape)

torch.save(x_train_NC_EMCI, "x_train_NC_EMCI.pt")
torch.save(x_test_NC_EMCI, "x_test_NC_EMCI.pt")
torch.save(y_train_NC_EMCI, "y_train_NC_EMCI.pt")
torch.save(y_test_NC_EMCI, "y_test_NC_EMCI.pt")