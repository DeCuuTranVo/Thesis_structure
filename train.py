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


def train(): 
    seed_torch(seed=0)
    
    # Initialisation
    best_score = 0	
    best_test_loss = float('inf') # Initial best validation loss    
    nonimproved_epoch = 0
    # Load parameters
    params = json.load(open('./config/config_train.json', 'r'))
    pprint(params)

    # LOAD TRAINER
    trainer = BaselineClassifier(**params)
    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.set_up_training_data(train_or_val = "train")
    # Set up training params
    model, optimizer, loss_criteria, lr_scheduler, device = trainer.set_up_training()

    # print("cuda",trainer.CUDA)

    # Train over 10 epochs (We restrict to 10 for time issues)
    # print('Training on', device)
    for epoch in range(1, trainer.EPOCHS + 1):
        train_loss, train_acc, train_auc = trainer.epoch_train(model, device, train_dataloader, optimizer, loss_criteria, epoch)
        test_loss, test_acc, test_auc = trainer.epoch_evaluate(model, device,  test_dataloader, loss_criteria, epoch)
        trainer.epoch_nums.append(epoch)
        trainer.training_loss.append(train_loss)
        trainer.validation_loss.append(test_loss)
        trainer.training_accuracy.append(train_acc)
        trainer.validation_accuracy.append(test_acc)     
        
        # Update learning rate (according to monitoring metrics)
        if trainer.MONITOR == "val_loss":
            lr_scheduler.step(test_loss)
        elif trainer.MONITOR == "val_acc":
            lr_scheduler.step(test_acc)
        elif trainer.MONITOR == "val_auc":
            lr_scheduler.step(test_auc)
        elif trainer.MONITOR == "train_loss":
            lr_scheduler.step(train_loss)
        elif trainer.MONITOR == "train_acc":
            lr_scheduler.step(train_acc)
        elif trainer.MONITOR == "train_auc":
            lr_scheduler.step(train_auc)
        else:
            raise ValueError("Wrong monitor arguments, only 3 options:'val_loss', 'val_acc', 'val_auc'")
            # Save model
            
        if not os.path.exists(trainer.MODEL_DIR):
            os.makedirs(trainer.MODEL_DIR)

        if trainer.MONITOR == "val_acc":
            if best_score <= test_acc:
                print('Improve accuracy from {:.4f} to {:.4f}'.format(best_score, test_acc))
                best_score = test_acc
                nonimproved_epoch = 0
                torch.save(model.state_dict(), trainer.saved_model_path) # Make sure the folder `models/` exists
                print("Save model to {:s}".format(trainer.saved_model_path))
                
                with open(trainer.OUTPUT_FILE, 'a') as f:
                    print('Improve accuracy from {:.4f} to {:.4f}'.format(best_score, test_acc), file=f)
                    print("Save model to {:s}".format(trainer.saved_model_path), file=f)

            else:
                nonimproved_epoch += 1
                print("Model not improving {:d}/{:d} epochs. Max accuracy {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_score))
                
                with open(trainer.OUTPUT_FILE, 'a') as f:
                    print("Model not improving {:d}/{:d} epochs. Max accuracy {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_score), file=f)

        elif trainer.MONITOR == "val_loss":
            if test_loss <= best_test_loss:
                print('Reduce loss from {:.4f} to {:.4f}'.format(best_test_loss, test_loss))
                best_test_loss = test_loss
                nonimproved_epoch = 0
                torch.save(model.state_dict(), trainer.saved_model_path)
                print("Save model to {:s}".format(trainer.saved_model_path))
                
                with open(trainer.OUTPUT_FILE, 'a') as f:
                    print('Reduce loss from {:.4f} to {:.4f}'.format(best_test_loss, test_loss), file=f)
                    print("Save model to {:s}".format(trainer.saved_model_path), file=f)
            else:
                nonimproved_epoch += 1
                print("Model not improving {:d}/{:d} epochs. Min loss {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_test_loss))
        
                with open(trainer.OUTPUT_FILE, 'a') as f:
                    print("Model not improving {:d}/{:d} epochs. Min loss {:.4f}".format(nonimproved_epoch,trainer.PATIENCE, best_test_loss), file=f)
        
        if nonimproved_epoch > trainer.PATIENCE:
            print('Early stopping!!!')
            break
        
        # if time.time() - start_time > trainer.TRAINING_TIME_OUT:
        # 	print('Early stopping. Out of time.')


    # torch.save(model.state_dict(), trainer.saved_model_path)   
    # print('model saved at', trainer.saved_model_path)

    print('################################################')
    
    


if __name__ == '__main__':
    start_time = timeit.default_timer()
    train()
    stop_time = timeit.default_timer()
    print("Time:", stop_time - start_time)
    
    # torch.save({"model": model.state_dict(),
	# 					"optimizer": optimizer.state_dict(),
	# 					"best_score": best_score,
	# 					"epoch": epoch,
	# 					"lr_scheduler": lr_scheduler.state_dict()}, 'models/retina_epoch{}_score{:.4f}.pth'.format(epoch, new_score)) # Make sure the folder `models/` exists
	