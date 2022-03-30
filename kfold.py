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
from src.model import NeuralNetwork
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import resnet_3d
import json
from pprint import pprint
import time
from src.trainer import BaselineClassifier, Trainer
from efficientnet_pytorch_3d import EfficientNet3D
import random
from torch.utils.tensorboard import SummaryWriter
from src.utils import seed_torch
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validate(): 
    # Load parameters
    params = json.load(open('./config/config_cv.json', 'r'))
    print(params)
    

    # LOAD TRAINER
    trainer = BaselineClassifier(**params)
    # print training parameters
    with open(trainer.OUTPUT_FILE, 'a') as f: 
        print(params, file=f)
        
    # set random state
    seed_torch(seed=trainer.SEED)
    
    trainer.load_training_data_cross_validation()
    
    # cross validation split
    k_folds = 5
    inner_stkfold = StratifiedKFold(n_splits=k_folds, shuffle=trainer.SHUFFLE_DATALOADER)
    
    best_score_kfold = []
    best_test_loss_kfold = []
    
    for fold, (train_ids, val_ids) in enumerate(inner_stkfold.split(trainer.x_cross_val, trainer.y_cross_val)):
        # Print
        print(f'FOLD {fold}')
        print("TRAIN:", train_ids, "VAL:", val_ids)
        # print("type of train_ids",type(train_ids))
        # print("type of val_ids",type(val_ids))
        print('--------------------------------')
        
        fold_epoch_nums = []
        fold_training_loss = []
        fold_validation_loss = []
        fold_training_accuracy = []
        fold_validation_accuracy = []
        fold_training_area_under_curve = []
        fold_validation_area_under_curve = []
        fold_learning_rate_history = []
        
        fold_best_metrics = {
            "epoch": None,
            "train_loss": None,
            "validation_loss": None,
            "train_accuracy": None,
            "validation_accuracy": None,
            "train_area_under_curve": None,
            "validation_area_under_curve": None
        }
        
        trainloader, valloader = trainer.set_up_training_data_cross_validation(train_ids, val_ids, fold)
        print(trainloader, valloader)
        
        # Set up training params
        model, optimizer, loss_criteria, lr_scheduler, device = trainer.set_up_training()
    
        # Initialisation
        best_score = 0	
        best_test_loss = float('inf') # Initial best validation loss    
        nonimproved_epoch = 0
    
        # print("cuda",trainer.CUDA)

        # Train over 10 epochs (We restrict to 10 for time issues)
        # print('Training on', device)
        for epoch in range(1, trainer.EPOCHS + 1):
            train_loss, train_acc, train_auc = trainer.epoch_train(model, device, trainloader, optimizer, loss_criteria, epoch)
            val_loss, val_acc, val_auc = trainer.epoch_evaluate(model, device, valloader, loss_criteria, epoch)
            
            fold_epoch_nums.append(epoch)          
            fold_training_loss.append(train_loss)
            fold_validation_loss.append(val_loss)    
            fold_training_accuracy.append(train_acc)          
            fold_validation_accuracy.append(val_acc)      
            fold_training_area_under_curve.append(train_auc)
            fold_validation_area_under_curve.append(val_auc)   
          
            if trainer.SCHEDULER == "reduce":
                # Update learning rate (according to monitoring metrics)
                if trainer.MONITOR == "val_loss":
                    lr_scheduler.step(val_loss)
                elif trainer.MONITOR == "val_acc":
                    lr_scheduler.step(val_acc)
                elif trainer.MONITOR == "val_auc":
                    lr_scheduler.step(val_auc)
                elif trainer.MONITOR == "train_loss":
                    lr_scheduler.step(train_loss)
                elif trainer.MONITOR == "train_acc":
                    lr_scheduler.step(train_acc)
                elif trainer.MONITOR == "train_auc":
                    lr_scheduler.step(train_auc)
                else:
                    raise ValueError("Wrong monitor arguments, only 3 options:'val_loss', 'val_acc', 'val_auc'")
                    # Save model
                current_learning_rate = optimizer.param_groups[0]['lr']  
            elif trainer.SCHEDULER == "cyclic":
                lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                # current_learning_rate = lr_scheduler.get_last_lr()[0]
            else:
                raise ValueError("Wrong SCHEDULER")
        
            fold_learning_rate_history.append(current_learning_rate)
            trainer.writer.add_scalar("Learning rate history", current_learning_rate,  epoch+ trainer.START_RECORD_EPOCH)
            with open(trainer.OUTPUT_FILE, 'a') as f: 
                print("Learning rate of epoch: {} is {}".format(epoch,current_learning_rate), file=f)            
                
            if not os.path.exists(trainer.MODEL_DIR):
                os.makedirs(trainer.MODEL_DIR)

            if trainer.MONITOR == "val_acc":
                if best_score <= val_acc:
                    print('Improve accuracy from {:.4f} to {:.4f}'.format(best_score, val_acc))
                    best_score = val_acc
                    nonimproved_epoch = 0
                    torch.save(model.state_dict(), trainer.saved_model_path) # Make sure the folder `models/` exists
                    print("Save model to {:s}".format(trainer.saved_model_path))
                    
                    with open(trainer.OUTPUT_FILE, 'a') as f:
                        print('Improve accuracy from {:.4f} to {:.4f}'.format(best_score, val_acc), file=f)
                        print("Save model to {:s}".format(trainer.saved_model_path), file=f)
                        
                    fold_best_metrics["epoch"] = epoch                    

                else:
                    nonimproved_epoch += 1
                    print("Model not improving {:d}/{:d} epochs. Max accuracy {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_score))
                    
                    with open(trainer.OUTPUT_FILE, 'a') as f:
                        print("Model not improving {:d}/{:d} epochs. Max accuracy {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_score), file=f)

            elif trainer.MONITOR == "val_loss":
                if val_loss <= best_test_loss:
                    print('Reduce loss from {:.4f} to {:.4f}'.format(best_test_loss, val_loss))
                    best_test_loss = val_loss
                    nonimproved_epoch = 0
                    torch.save(model.state_dict(), trainer.saved_model_path)
                    print("Save model to {:s}".format(trainer.saved_model_path))
                    
                    with open(trainer.OUTPUT_FILE, 'a') as f:
                        print('Reduce loss from {:.4f} to {:.4f}'.format(best_test_loss, val_loss), file=f)
                        print("Save model to {:s}".format(trainer.saved_model_path), file=f)
                        
                    fold_best_metrics["epoch"] = epoch 
                    
                else:
                    nonimproved_epoch += 1
                    print("Model not improving {:d}/{:d} epochs. Min loss {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_test_loss))
            
                    with open(trainer.OUTPUT_FILE, 'a') as f:
                        print("Model not improving {:d}/{:d} epochs. Min loss {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_test_loss), file=f)
            
            if nonimproved_epoch > trainer.PATIENCE:
                print('Early stopping!!!')
                break


        # torch.save(model.state_dict(), trainer.saved_model_path)   
        # print('model saved at', trainer.saved_model_path)
        
        
        trainer.epoch_nums.append(fold_epoch_nums)
        trainer.training_loss.append(fold_training_loss)
        trainer.validation_loss.append(fold_validation_loss)
        trainer.training_accuracy.append(fold_training_accuracy)
        trainer.validation_accuracy.append(fold_validation_accuracy)
        trainer.training_area_under_curve.append(fold_training_area_under_curve)
        trainer.validation_area_under_curve.append(fold_validation_area_under_curve)
        trainer.learning_rate_history.append(fold_learning_rate_history)
        print('################################################')
        
        del model, optimizer, loss_criteria, lr_scheduler, device, trainloader, valloader
        
        best_score_kfold.append(best_score)
        best_test_loss_kfold.append(best_test_loss)
        
        best_index = fold_best_metrics["epoch"] - 1

        fold_best_metrics["train_loss"] = fold_training_loss[best_index]
        fold_best_metrics["validation_loss"] = fold_validation_loss[best_index]
        fold_best_metrics["train_accuracy"] = fold_training_accuracy[best_index]
        fold_best_metrics["validation_accuracy"] = fold_validation_accuracy[best_index]
        fold_best_metrics["train_area_under_curve"] = fold_training_area_under_curve[best_index]
        fold_best_metrics["validation_area_under_curve"] = fold_validation_area_under_curve[best_index]    
        
        trainer.cross_val_best_metrics.append(fold_best_metrics)
        
    
    print('################################################ END CROSS VALIDATION ########################')
    print("best score kfold",best_score_kfold)
    print("best test loss kfold",best_test_loss_kfold)
    print("trained model saved at",trainer.saved_model_path)
    print("cross validation best metrics summary", trainer.cross_val_best_metrics)


if __name__ == '__main__':
    start_time = timeit.default_timer()
    cross_validate()
    stop_time = timeit.default_timer()
    print("Time:", stop_time - start_time)

