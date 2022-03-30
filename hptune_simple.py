from functools import partial
from operator import mod
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray
import json
from pprint import pprint
from src.trainer import BaselineClassifier
from src.utils import seed_torch, calculate_metrics
from src.model import NeuralNetwork
from ray.tune.suggest.bayesopt import BayesOptSearch
from src.dataset import CustomImageDataset
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
import json

def load_data(data_dir="./tensor"):
    x_train = torch.load(os.path.join(data_dir,"x_train_tensor.pt"))
    x_test = torch.load(os.path.join(data_dir,"x_test_tensor.pt"))
    y_train = torch.load(os.path.join(data_dir,"y_train_tensor.pt"))
    y_test = torch.load(os.path.join(data_dir,"y_test_tensor.pt"))
    
    x_train = torch.unsqueeze(x_train, 1)
    x_test = torch.unsqueeze(x_test, 1)
    y_train = y_train.type(torch.long) #long
    y_test = y_test.type(torch.long) #long
    
    train_dataset = CustomImageDataset(
            x_train,
            y_train,
            augment=True,
            train_or_val="train",
            rotation=(-5,5),
            translation=(0,0.1),
            scaling=(0.8,1.2))  # augment = True
    test_dataset = CustomImageDataset(
            x_test,
            y_test,
            train_or_val = "val",
            augment=False)

    return train_dataset, test_dataset

def train_adni(config, checkpoint_dir=None, data_dir=None, trainer=None):   
    model = NeuralNetwork("shufflenet_v2", num_classes=4, is_pretrained=True, dropout_rate=config["dropout"])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    trainset, testset = load_data(data_dir)
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
    
    train_dataloader = torch.utils.data.DataLoader(train_subset, 
                batch_size=config["batch_size"],
                num_workers=16,
                shuffle=True,
                sampler = None
            ) # create your dataloader

    val_dataloader = torch.utils.data.DataLoader(val_subset, 
                batch_size=config["batch_size"],
                num_workers=16,
                shuffle=True,
                sampler = None
            ) # create your dataloader
    
    
    for epoch in range(10):  # loop over the dataset multiple times #10
        
        ##### Training part #################################
        model.train()        
        train_loss = 0
        train_correct = 0
        
        out_pred_train = torch.FloatTensor().to(device)
        out_gt_train = torch.FloatTensor().to(device)
        
        for batch_idx, (data, target) in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            data, target = data.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(data)
            loss = loss_criteria(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            
            # Print metrics so we see some progress
            # print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            train_correct += torch.sum(target==predicted).item()
            
            # Update groundtruth values
            out_gt_train = torch.cat((out_gt_train, target), 0)
            
            output = torch.nn.functional.softmax(output, dim=1)
            # Update prediction values
            out_pred_train = torch.cat((out_pred_train, output), 0)
        
        # return average loss for the epoch
        avg_loss_train = train_loss / (batch_idx+1)
        accuracy_train, _, _, _, _, _, auc_score_train = calculate_metrics(out_gt_train, out_pred_train)
        print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            avg_loss_train, 100 * accuracy_train, auc_score_train))
        
        
        ##### Validation part #################################
        # Total torch tensor of all prediction and groundtruth
        out_pred_val = torch.FloatTensor().to(device)
        out_gt_val = torch.FloatTensor().to(device)
        # Validation loss
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            batch_count = 0
            for data, target in val_dataloader:
                batch_count += 1
                data, target = data.to(device), target.to(device)
                
                # Get the predicted classes for this batch
                output = model(data)
                
                # Calculate the loss for this batch
                val_loss += loss_criteria(output, target).item()
                
                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                val_correct += torch.sum(target==predicted).item()
                
                # Update groundtruth values
                out_gt_val = torch.cat((out_gt_val, target), 0)
                
                output = torch.nn.functional.softmax(output, dim=1)
                
                # Update prediction values
                out_pred_val = torch.cat((out_pred_val, output), 0)
        
        avg_loss_val = val_loss / batch_count
        accuracy_val, _, _, _, _, _, auc_score_val = calculate_metrics(
            out_gt_val, out_pred_val)
        
        print(
            'Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
                avg_loss_val, 
                100. * accuracy_val,
                auc_score_val))
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=avg_loss_val, accuracy=accuracy_val, area_under_curve = auc_score_val)
    print("Finished Training")

def test_accuracy(model, loss_criteria, device="cpu"):
    trainset, testset = load_data()
    
    test_dataloader = torch.utils.data.DataLoader(testset, 
                batch_size=4,
                num_workers=16,
                shuffle=False,
                sampler = None
            ) # create your dataloader
    
    
    # Total torch tensor of all prediction and groundtruth
    out_pred_test = torch.FloatTensor().to(device)
    out_gt_test = torch.FloatTensor().to(device)
    # Validation loss
    model.eval()
    test_loss = 0
    test_correct = 0
        
    with torch.no_grad():
        batch_count = 0
        for data, target in test_dataloader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            test_correct += torch.sum(target==predicted).item()
            
            # Update groundtruth values
            out_gt_test = torch.cat((out_gt_test, target), 0)
            
            output = torch.nn.functional.softmax(output, dim=1)
            
            # Update prediction values
            out_pred_test = torch.cat((out_pred_test, output), 0)
        
    avg_loss_test = test_loss / batch_count
    accuracy_test, _, _, _, _, _, auc_score_test = calculate_metrics(
        out_gt_test, out_pred_test)
        
    print(
        'Test set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            avg_loss_test, 
            100. * accuracy_test,
            auc_score_test))
        
    return accuracy_test
        
def main(num_samples, max_num_epochs, gpus_per_trial):
    seed_torch(seed=1234)
    data_dir = os.path.abspath("./tensor")
    train_dataset, test_dataset = load_data()
    # load_data()
    print(train_dataset)
    print(test_dataset)
    config = {
        "dropout": tune.uniform(0,1),
        "l2": tune.uniform(0,1),
        "lr": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([8, 16, 32, 64])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=30,
        reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=["dropout", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "area_under_curve","training_iteration"])
    
    # space_hyperopt = {        
    #     "dropout": hp.uniform("dropout", 0.,1),
    #     "l2": hp.uniform("l2", 0,1),
    #     "lr": hp.loguniform("lr",1e-6, 1e-3),
    #     "batch_size": hp.choice("batch_size",[2, 4, 8, 16])
    # }
    # hyperopt_search = HyperOptSearch(space_hyperopt, metric="mean_accuracy", mode="max")

    # space_bayesopt = {
    # "dropout": (0,1),
    # "l2": (0,1),
    # "lr": (1e-6, 1e-3),
    # "batch_size": ([2, 4, 8, 16])
    # }
    # bayesopt_search = BayesOptSearch( metric="mean_loss", mode="min")
    
    result = tune.run(
        partial(train_adni, data_dir=data_dir),
        # search_alg = hyperopt_search,
        # search_alg = bayesopt_search,
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation auroc: {}".format(
        best_trial.last_result["area_under_curve"]))
    
    print("best trial config ", config["dropout"])
    print("type of best_trial config: ", type(config))
    print("type of best_trial config dropout: ", type(config["dropout"]))
    
    best_trained_model = NeuralNetwork("shufflenet_v2", num_classes=4, is_pretrained=True, dropout_rate=float(config["dropout"].sample()))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    # Write best trial to json file
    # e.g. file = './data.json' 
    with open("output0.json", 'w') as f: 
        json.dump(best_trial.config, f)
    
    best_checkpoint_dir = best_trial.checkpoint.value
    print("best_Checkpoint_dir",best_checkpoint_dir)
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    loss_criteria = nn.CrossEntropyLoss()
    test_acc = test_accuracy(best_trained_model, loss_criteria,device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    # Get a dataframe for analyzing trial results.
    df = result.results_df
    df.to_csv('raytune_result_df.csv', index=False)
    
    
    
import time
if __name__ == "__main__":
    start_time = time.clock()
    # You can change the number of GPUs per trial here:
    main(num_samples=20, max_num_epochs=120, gpus_per_trial=2)
    
    print(time.clock() - start_time, "seconds")
    
    