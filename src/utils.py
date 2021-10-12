import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import random

class EarlyStopping:
    """Early stops the training if validation loss and validation accuracy don't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='models/checkpoint.pt', trace_func=print, monitor='val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print         
            monitor (Mode): If val_loss, stop at maximum mode, else val_accuracy, stop at minimum mode
                            Default: val_loss   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = monitor

    def __call__(self, values, model):

        if self.mode == 'val_loss':
            score = -values

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0
        else:
            score = values
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0

    def save_checkpoint(self, values, model):
        '''Saves model when validation loss decrease.'''
        if self.mode == 'val_loss':
            if self.verbose:
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {values:.6f}).   Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = values
        elif self.mode == 'val_accuracy':
            if self.verbose:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_acc_max:.3f} --> {values:.3f}).  Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = values


def preprocess(data_dir, csv_dir):
    """
    Get training dataframe and testing dataframe from image directory and
    csv description file.

    Args:
        data_dir (String): Directory of image data
        csv_dir (String): Directory of csv description file

    Returns:
        df_train (pandas.DataFrame): Data frame of training set
        df_test (pandas.DataFrame):  Data frame of test set
    """
    data_name = os.listdir(data_dir)
    url_dataframe = pd.read_csv(csv_dir)

    url_dataframe["ID"] = [str(x) + ".png" for x in url_dataframe["ID"]]
    url_dataframe["Label"] = [
        0 if x == "anterior" else 1 for x in url_dataframe["Label"]]

    total_name = url_dataframe["ID"]
    total_label = url_dataframe["Label"]

    name_train, name_test, label_train, label_test = train_test_split(
        total_name, total_label, test_size=0.3, random_state=42)

    data_train = {'Name': name_train,
                  'Label': label_train}

    data_test = {'Name': name_test,
                 'Label': label_test}

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    print("preprocessing complete")
    return df_train, df_test


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# print(np.array(y_pred.cpu()))
# y_pred_1hot = to_categorical(y_pred.cpu(), num_classes=4)
# y_test_1hot = to_categorical(y_test.cpu(), num_classes=4) #y_test***
# print(y_pred_1hot)
# print(y_test_1hot)

def calculate_metrics(out_gt, out_pred):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity

    """
    
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    
    out_gt = to_categorical(out_gt.astype('uint8'), num_classes=4)
    try:
        auc_score = roc_auc_score(out_gt, out_pred, average="macro")
    except:
        auc_score = 0
    out_pred = np.argmax(out_pred, axis=1)
    out_pred = to_categorical(out_pred.astype('uint8'), num_classes=4)
    
    # # print("out_gt_metrics", out_gt)
    # print("shape of out_gt_metrics",out_gt.shape)
    # # # print(out_gt.dtype)
    # # print("out_pred_metrics",out_pred)
    # print("shape of out_pred_metrics",out_pred.shape)
    # # # print(out_pred.dtype)
    
    accuracy = accuracy_score(out_gt, out_pred)
    precision = precision_score(out_gt, out_pred, average = "macro", zero_division=0)
    recall = recall_score(out_gt, out_pred, average = "macro", zero_division=0)
    F1_score = f1_score(out_gt, out_pred, average = "macro", zero_division=0)
    
    sensitivity = recall
    specificity = 0

    return accuracy, precision, recall, F1_score, sensitivity, specificity, auc_score

from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sns
def create_confusion_matrix(out_gt, out_pred, problem):
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    
    # out_gt = to_categorical(out_gt.astype('uint8'), num_classes=4)

    out_pred = np.argmax(out_pred, axis=1)
    # out_pred = to_categorical(out_pred.astype('uint8'), num_classes=4)
    
    # print(out_gt)
    # print(out_pred)
    print(np.unique(out_pred, axis=0))
    
    cm = confusion_matrix(out_gt , out_pred)
    print(cm)
    print([i for i in np.unique(out_pred, axis=0)])
    
    if problem == "four_classes":
        label_list = ['NC', 'EMCI', 'LMCI', 'AD']
        index_list = [0,1,2,3]
        columns_list = [0,1,2,3]
    elif problem == "NC_AD":
        label_list = ['NC', 'AD']
        index_list = [0,1]
        columns_list = [0,1]
    elif problem == "NC_EMCI":
        label_list = ['NC', 'EMCI']
        index_list = [0,1]
        columns_list = [0,1]
    else:
        print("wrong problem")
    cm = pd.DataFrame(cm , index = index_list , columns = columns_list)
    # print(cm)
    # ######################################################################################
    cm_figure = plt.figure(figsize = (10,8))    
    cm_heatmap = sns.heatmap(cm, linecolor = 'black' , linewidth = 0.5 , annot = True, fmt='d', xticklabels = label_list, yticklabels =label_list  ) 
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel("Prediction", fontsize=18)
    plt.ylabel("Ground Truth", fontsize=18)
    plt.show()
    # plt.imshow(cm_heatmap)
    plt.savefig("confusion_matrix.png")
    return cm

def create_classification_report(out_gt, out_pred, problem):
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    
    

    out_pred = np.argmax(out_pred, axis=1)
    
    
    if problem == "four_classes":
        # CLASSIFICATION REPORT
        classes = {          
                2 :('LMCI', 'late cognitive impairment'), 
                1:('EMCI' , 'early cognitive impairment'), 
                0: ('NC', 'normal control'),  
                3: ('AD', 'alzheimers disease')}
        out_gt = to_categorical(out_gt.astype('uint8'), num_classes=4)
        out_pred = to_categorical(out_pred.astype('uint8'), num_classes=4)
    elif problem == "NC_AD":
                # CLASSIFICATION REPORT
        classes = {          
                0: ('NC', 'normal control'),  
                1: ('AD', 'alzheimers disease')}
        out_gt = to_categorical(out_gt.astype('uint8'), num_classes=2)
        out_pred = to_categorical(out_pred.astype('uint8'), num_classes=2)
    elif problem == "NC_EMCI":
        classes = {          
                1:('EMCI' , 'early cognitive impairment'), 
                0: ('NC', 'normal control')}
        out_gt = to_categorical(out_gt.astype('uint8'), num_classes=2)
        out_pred = to_categorical(out_pred.astype('uint8'), num_classes=2)
    else: 
        print("wrong problem")

    print(problem)
    print(classes)
    print(range(len(classes)))
    
    target_names = [f"{classes[i]}" for i in range(len(classes))]
    print(target_names)
    class_report = classification_report(out_gt, out_pred , target_names =target_names )
    return class_report

 
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    preprocess('../data', '../url_and_label.csv')
