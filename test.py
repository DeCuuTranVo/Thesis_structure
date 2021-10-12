import torch
import json
import os
from src.trainer import BaselineClassifier
from src.utils import seed_torch


def test():
    """
    Test the model with configuration loaded from json file
    """
    seed_torch(seed=0)

    # Load testing parameters

    params = json.load(open('config/config_test.json', 'r'))

    # params_string = json.dumps(params, indent= 4, separators= ", ")
    # print(params_string)

    # Create CustomTrainer instance with loaded testing parameters
    trainer = BaselineClassifier(**params)

    # print(trainer.__dict__)

    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.set_up_training_data(train_or_val = "val") #"val"
    # print("line 28", len(test_dataloader.dataset))
    # Set up training params
    model, _, loss_criteria, _, device = trainer.set_up_training()

    # Load trained model
    # model.load_state_dict(torch.load(os.path.join(
    #     trainer.MODEL_DIR, "trial-" + trainer.TRIAL + ".pth")))

    # Calculate test loss and accuracy
    train_loss, train_acc, train_auc = trainer.epoch_evaluate(model, device, train_dataloader, loss_criteria, train_or_validation="validation")

    # Calculate test loss and accuracy
    test_loss, test_acc, test_auc = trainer.epoch_evaluate(model, device,  test_dataloader, loss_criteria, train_or_validation="validation")
    print("train_loss",train_loss)
    print("train_acc",train_acc)
    print("train_auc", train_auc)
    print("test_loss",test_loss)
    print("test_acc",test_acc)
    print("test_auc", test_auc)
if __name__ == '__main__':
    test()
