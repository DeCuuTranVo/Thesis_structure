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
from src.utils import seed_torch
from src.model import NeuralNetwork
from ray.tune.suggest.bayesopt import BayesOptSearch

def train_adni(config, checkpoint_dir=None, data_dir=None, trainer=None):    
    # net = Net(config["l1"], config["l2"])
    # Set up training params
    trainer.LEARNING_RATE_START = config["lr"]
    model, optimizer, loss_criteria, lr_scheduler, device = trainer.set_up_training()
    

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    
    # Set up DataLoader
    trainer.BATCH_SIZE = int(config['batch_size'])
    train_dataloader, test_dataloader = trainer.set_up_training_data(train_or_val = "train")
    #### VALSET IS THE SAME AS TESTSET IN MAIN -> NEED CHANGE IN FUTURE!!!
    for epoch in range(1, trainer.EPOCHS + 1):  # loop over the dataset multiple times
        train_loss, train_acc, train_auc = trainer.epoch_train(model, device, train_dataloader, optimizer, loss_criteria, epoch)
        test_loss, test_acc, test_auc = trainer.epoch_evaluate(model, device,  test_dataloader, loss_criteria, epoch)
        lr_scheduler.step(test_acc)

        ### IMPORTANT
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=test_loss, accuracy=test_acc)
    print("Finished Training")
    
    
def test_accuracy(net, device="cpu"):
    return 0
        
        
        
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    # Load parameters
    params = json.load(open('./config/config_train.json', 'r'))
    pprint(params)

    # LOAD TRAINER
    trainer = BaselineClassifier(**params)
    seed_torch(seed=trainer.SEED)
    
    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.set_up_training_data(train_or_val = "train")
    
    #???
    # Set up training params
    model, optimizer, loss_criteria, lr_scheduler, device = trainer.set_up_training()
    
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    
    
    bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
    print(bayesopt)
    # print(config)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    # print(scheduler)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    print(reporter)
    
    result = tune.run(
        partial(train_adni, trainer=trainer),
        # train_adni,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        # search_alg = bayesopt,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    best_trained_model = NeuralNetwork(trainer.ARCHITECTURE, num_classes=trainer.NUM_CLASSES, is_pretrained=True, dropout_rate=trainer.DROPOUT_RATE)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    test_loss, test_acc, test_auc = trainer.epoch_evaluate(model, device,  test_dataloader, loss_criteria, epoch=0)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=2)
    
    