import os, sys
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import numpy as np
from collections import Counter
import torch
from torchsummary import summary
from efficientnet_pytorch_3d import EfficientNet3D
import torch
import timeit
import gc
from numpy import savez_compressed




base_path = '/mnt/data_lab513/vqtran_data'
root_data = os.path.join(base_path, "data", "raw_data", "ADNI_NIfTI")
root_bias_correction = os.path.join(base_path, "data", "clean_data", "mri_bias_correction")
root_bet = os.path.join(base_path, "data", "clean_data", "mri_brain_extraction")
root_reg = os.path.join(base_path, "data", "clean_data", "mri_registration")
root_meta = os.path.join(base_path, "data", "meta_data")#, "Pre-Thesis_metadata", "ADNI") 
root_train = os.path.join(base_path, "data", "train_data")
root_numpy_array = os.path.join(base_path, "data", 'data_numpy_array')
root_model = os.path.join(base_path, 'Model')



TARGET_WIDTH = 110
TARGET_HEIGHT = 110
TARGET_DEPTH = 110
BATCH_SIZE = 2
BUFFER_SIZE = 1
initial_learning_rate =   0.001 #0.000027 

start_time = timeit.default_timer()

# # load numpy array from npz file
from numpy import load



dict_x_train = load(os.path.join(base_path, "data",  'data_numpy_array', 'x_train_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH) +'.npz'))
dict_x_val = load(os.path.join(base_path, "data",  'data_numpy_array', 'x_val_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH) +'.npz'))
dict_x_test = load(os.path.join(base_path, "data",   'data_numpy_array', 'x_test_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH)  +'.npz'))
dict_y_train = load(os.path.join(base_path, "data",  'data_numpy_array', 'y_train_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH)  +'.npz'))
dict_y_val = load(os.path.join(base_path, "data",   'data_numpy_array', 'y_val_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH) +'.npz'))
dict_y_test = load(os.path.join(base_path, "data",   'data_numpy_array', 'y_test_' + str(TARGET_WIDTH) + str(TARGET_HEIGHT) + str(TARGET_DEPTH) +'.npz'))

x_train = dict_x_train['arr_0']
x_val = dict_x_val['arr_0']
x_test = dict_x_test['arr_0']
y_train = dict_y_train['arr_0']
y_val= dict_y_val['arr_0']
y_test = dict_y_test['arr_0']
        
    # del dict_x_train, dict_y_train
    # # del dict_x_train, dict_x_test, dict_x_val, dict_y_train, dict_y_test, dict_y_val
    # # gc.collect()
# print the array
# print(x_train.shape)
# print(x_val.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)


# np.save('x_train.npy', x_train)
# savez_compressed('x_train.npy', x_train)
# np.save("x_val.npy", x_val)
# np.save("x_test.npy", x_test)
# np.save("y_train.npy", y_train)
# np.save("y_val.npy", y_val)
# np.save("y_test.npy", y_test)

# gc.collect()
x_train = np.append(x_train, x_val, axis=0)
# del x_val
y_train = np.append(y_train, y_val, axis=0)
# del y_val
# # print the array
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# # np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
x_train_tensor = torch.from_numpy(x_train)
# del x_train
x_test_tensor = torch.from_numpy(x_test)
# del x_test
y_train_tensor = torch.from_numpy(y_train)
# del y_train
y_test_tensor = torch.from_numpy(y_test)
# del y_test


torch.save(x_train_tensor, "x_train_tensor.pt")
torch.save(x_test_tensor, "x_test_tensor.pt")
torch.save(y_train_tensor, "y_train_tensor.pt")
torch.save(y_test_tensor, "y_test_tensor.pt")


print("save complete")
stop_time = timeit.default_timer()
print("Time:", stop_time-start_time, "seconds")