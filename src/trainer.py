import torch.optim as optim
import torch.nn as nn
import torch
from torchsummary import summary
from src.model import NeuralNetwork
from src.utils import calculate_metrics, create_confusion_matrix, create_classification_report
import os
from src.dataset import CustomImageDataset
from torchsampler import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key.upper(), args[key])
   

class BaselineClassifier(Trainer):
    def __init__(self, **args):
        super(BaselineClassifier, self).__init__(**args)
        
        # Track metrics in these arrays
        self.epoch_nums = []
        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        
        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter(os.path.join(self.LOG_DIR, "trial_" + str(self.TRIAL)))  #'runs/resnet_adam_110'
        print(os.path.join(self.LOG_DIR, "trial_" + str(self.TRIAL)))
        
        self.saved_model_path = os.path.join(self.MODEL_DIR,"trial_" + str(self.TRIAL) + ".pth")
        self.x_train = 0
        self.x_test = 0
        self.y_train = 0
        self.y_test = 0
        self.class_weight = []
        
        if self.CLASS_IMBALANCE == "oversampling":
            self.SHUFFLE_DATALOADER = False
    
    def epoch_train(self, model, device, train_loader, optimizer, loss_criteria, epoch): # Epoch training
        '''
        Training rountine in each epoch
        Returns:
            train_loss
        '''
        # Set the model to training mode
        model.train()
        train_loss = 0
        correct = 0
        
        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)
        print("Epoch:", epoch)
        with open(self.OUTPUT_FILE, 'a') as f:
            print("Epoch:", epoch, file =f)
            
        # Process the images in batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # Use the CPU or GPU as appropriate
            # Recall that GPU is optimized for the operations we are dealing with
            data, target = data.to(device), target.to(device)
            
            # Reset the optimizer
            optimizer.zero_grad()
            
            # Push the data forward through the model layers
            output = model(data)
            
            # Get the loss
            loss = loss_criteria(output, target)

            # Keep a running total
            train_loss += loss.item()
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Print metrics so we see some progress
            print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()
            
            # Update groundtruth values
            out_gt = torch.cat((out_gt, target), 0)
            
            output = torch.nn.functional.softmax(output, dim=1)
            # Update prediction values
            out_pred = torch.cat((out_pred, output), 0)
                
        # return average loss for the epoch
        avg_loss = train_loss / (batch_idx+1)
        # print('Training set: Average loss: {:.6f}'.format(avg_loss))
        # print('Training set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     correct, len(train_loader.dataset),
        #     100. * correct / len(train_loader.dataset)))
        # print("out_gt_train shape", out_gt.shape)
        # print("out_pred_train shape", out_pred.shape)
        
        accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score = calculate_metrics(out_gt, out_pred)
        # with open("output.txt", "a") as f:        
        #     print("Epoch:", epoch, file =f)
        #     print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
        #     train_loss / (batch + 1), 100 * accuracy, auc_score), file =f)

        print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            train_loss / (batch_idx + 1), 100 * accuracy, auc_score))
        self.writer.add_scalar("Loss/train", train_loss / (batch_idx + 1), epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("Accuracy/train", accuracy,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("Precision/train", precision,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("Recall/train", recall,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("F1_score/train", f1_score,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("Sensitivity/train", sensitivity,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("Specificity/train", specificity,  epoch+self.START_RECORD_EPOCH)
        self.writer.add_scalar("AUROC/train", auc_score,  epoch+self.START_RECORD_EPOCH)

        with open(self.OUTPUT_FILE, 'a') as f:
            print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            train_loss / (batch_idx + 1), 100 * accuracy, auc_score), file=f)

        return avg_loss, accuracy , auc_score#correct / len(train_loader.dataset)

    def epoch_evaluate(self, model, device, test_loader, loss_criteria, epoch=0, train_or_validation="train"): # Epoch evaluating
        '''
        Validating model after each epoch
        Returns:
            val_loss
            val_f1_score
        '''
        # Calculate number of samples
        size = len(test_loader.dataset)
        # print("line 125", size)
        # Calculate number of batches
        num_batches = len(test_loader)
        
        # Total torch tensor of all prediction and groundtruth
        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)
 
        # Switch the model to evaluation mode (so we don't backpropagate or drop)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            batch_count = 0
            for data, target in test_loader:
                batch_count += 1
                data, target = data.to(device), target.to(device)
                
                # Get the predicted classes for this batch
                output = model(data)
                
                # Calculate the loss for this batch
                test_loss += loss_criteria(output, target).item()
                
                # Calculate the accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(target==predicted).item()
                
                # Update groundtruth values
                out_gt = torch.cat((out_gt, target), 0)
                
                output = torch.nn.functional.softmax(output, dim=1)
                
                # Update prediction values
                out_pred = torch.cat((out_pred, output), 0)

        # Calculate the average loss and total accuracy for this epoch
        avg_loss = test_loss / batch_count
        # print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avg_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))
        
        accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score = calculate_metrics(
            out_gt, out_pred)
        
        # print("out_gt_test shape", out_gt.shape)
        # print("out_pred_test shape", out_pred.shape)
        # with open("output.txt", "a") as f:
        #     print(
        #     'Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
        #         avg_loss, 
        #         100. * accuracy,
        #         auc_score), file = f)

        print(
            'Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
                avg_loss, 
                100. * accuracy,
                auc_score))
        
        if train_or_validation == "train":
            # TensorBoard
            self.writer.add_scalar("Loss/test", test_loss,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Accuracy/test", accuracy,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Precision/test", precision,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Recall/test", recall,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("F1_score/test", f1_score,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Sensitivity/test", sensitivity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Specificity/test", specificity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("AUROC/test", auc_score,  epoch+self.START_RECORD_EPOCH)
            
            with open(self.OUTPUT_FILE, 'a') as f:
                print('Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
                    avg_loss, 
                    100. * accuracy,
                    auc_score), file=f)
        elif train_or_validation == "validation":
            # print("line 194",len(out_gt))
            con_mat = create_confusion_matrix(out_gt, out_pred, self.PROBLEM)
            print(con_mat)
            
            class_report = create_classification_report(out_gt, out_pred, self.PROBLEM)
            print(class_report)
        else:
            print("Invalid training or validation")
        # return average loss for the epoch
        return avg_loss, accuracy, auc_score #correct / len(test_loader.dataset) # accuracy


    def set_up_training_data(self, train_or_val = "train"):
        '''
        Return: 
            train_dataloader, val_dataloader
        '''
        print('----- Setting up data ... -----')
        self.x_train = torch.load(self.X_TRAIN_PATH)
        self.x_test = torch.load(self.X_TEST_PATH)
        self.y_train = torch.load(self.Y_TRAIN_PATH)
        self.y_test = torch.load(self.Y_TEST_PATH)

        print(self.x_train.shape)
        print(self.x_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)

        print("load complete")


        self.x_train = torch.unsqueeze(self.x_train, 1)
        self.x_test = torch.unsqueeze(self.x_test, 1)
        self.y_train = self.y_train.type(torch.long) #long
        self.y_test = self.y_test.type(torch.long) #long
        
        # train_dataset = torch.utils.data.TensorDataset(self.x_train,self.y_train) # create your datset
        # test_dataset = torch.utils.data.TensorDataset(self.x_test,self.y_test)

        train_dataset = CustomImageDataset(
            self.x_train,
            self.y_train,
            augment=self.USE_AUGMENTATION,
            rotation=self.ROTATE,
            translation=self.TRANSLATE,
            scaling=self.SCALE)  # augment = True
        test_dataset = CustomImageDataset(
            self.x_test,
            self.y_test,
            augment=False)


        if self.DEBUG_DATASET:        
            train_dataset = torch.utils.data.Subset(train_dataset, range(50)) ### Delete this to use full dataset
            test_dataset = torch.utils.data.Subset(test_dataset, range(50)) ###

        
        distinct_label_train, counts_label_train =  torch.unique(self.y_train, return_counts=True)
        distinct_label_test, counts_label_test = torch.unique(self.y_test, return_counts=True)

        num_label_total = len(self.y_train) + len(self.y_test)
        # print(num_label_total)

        for label in distinct_label_train.tolist():
            self.class_weight.append(1.0/(counts_label_train[label]+counts_label_test[label])) #float(num_label_total)
        
        self.class_weight = torch.FloatTensor(self.class_weight)
        # print(self.class_weight)
        ###
        # if train_or_val == "train":
        #     train_dataloader = torch.utils.data.DataLoader(train_dataset, 
        #         batch_size=self.BATCH_SIZE,
        #         num_workers=self.NUM_WORKERS,
        #         shuffle=self.SHUFFLE_DATALOADER,
        #         sampler = torch.utils.data.sampler.WeightedRandomSampler(self.class_weight, len(self.class_weight)) 
        #     ) # create your dataloader

        #     test_dataloader = torch.utils.data.DataLoader(test_dataset, 
        #         batch_size=self.BATCH_SIZE,
        #         num_workers=self.NUM_WORKERS,
        #         shuffle=self.SHUFFLE_DATALOADER,
        #         sampler = torch.utils.data.sampler.WeightedRandomSampler(self.class_weight, len(self.class_weight)) 
        #     ) # create your dataloader
        # elif train_or_val == "val":
        # index_sequence = []
        weight_sequence_train = []
        weight_sequence_test = []
        

        for i in range(len(self.y_train)):
            for item in distinct_label_train.tolist():
                if self.y_train[i] == item:
                    # index_sequence.append(i)
                    weight_sequence_train.append(self.class_weight[item])
        
        for i in range(len(self.y_test)):
            for item in distinct_label_test.tolist():
                if self.y_test[i] == item:
                    # index_sequence.append(i)
                    weight_sequence_test.append(self.class_weight[item])
        
        if train_or_val == "train":
            if self.CLASS_IMBALANCE == "oversampling":
                print("Using oversampling")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_sequence_train, int(counts_label_train[0]*4)) #None
                ) # create your dataloader
                test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = None) #torch.utils.data.sampler.WeightedRandomSampler(weight_sequence_test, int(counts_label_test[0]*4))) #None
            elif self.CLASS_IMBALANCE == "weighted_loss":
                print("using weighted_loss")
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = None
                ) # create your dataloader
                test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = None) #torch.utils.data.sampler.WeightedRandomSampler(weight_sequence_test, int(counts_label_test[0]*4))) #None
            elif self.CLASS_IMBALANCE is None:
                train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = None
                ) # create your dataloader
                test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    shuffle=self.SHUFFLE_DATALOADER,
                    sampler = None) #torch.utils.data.sampler.WeightedRandomSampler(weight_sequence_test, int(counts_label_test[0]*4))) #None
            else:     
                raise ValueError("Wrong choice of class imbalance: only 3 options: \"oversampling\" or \"weighted_loss\" or None")
        
        elif train_or_val == "val":
            # self.SHUFFLE_DATALOADER = False
            train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                batch_size=self.BATCH_SIZE,
                num_workers=self.NUM_WORKERS,
                shuffle=self.SHUFFLE_DATALOADER,
                sampler = None
            ) # create your dataloader

            test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                batch_size=self.BATCH_SIZE,
                num_workers=self.NUM_WORKERS,
                shuffle=self.SHUFFLE_DATALOADER,
                sampler = None
            ) # create your dataloader
        else:
            raise ValueError("Problem with sampler")
             
        print("number of sample in train dataloader",len(train_dataloader.dataset))
        print("number of sample in test dataloader",len(test_dataloader.dataset))
        print('Complete DataLoader')
        return train_dataloader, test_dataloader

    def set_up_training(self):
        '''
        Return:
            model, optimizer, criterion, scheduer, device
        '''
        model = NeuralNetwork(self.ARCHITECTURE, num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
        
        
        if self.USE_TRAINED_MODEL:
            if self.PATH_PRETRAINED_MODEL is None:
                if os.path.exists(self.saved_model_path):            
                    model.load_state_dict(torch.load(self.saved_model_path))
                    print("load trained model from", self.saved_model_path)
                else:
                    print("trained model",self.saved_model_path ,"not exist!")
                    print("training from scratch")
            else:
                model_dict = model.state_dict()
                pretrained_dict =  torch.load(self.PATH_PRETRAINED_MODEL)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                del model_dict, pretrained_dict
                print("load trained model from", self.PATH_PRETRAINED_MODEL)
        else:
            print("TRAINING FROM SCRATCH")

        if self.OPTIMIZER == "adam":
            # Use an "Adam" optimizer to adjust weights
            optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE_START, weight_decay=self.WEIGHT_DECAY)
        elif self.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.LEARNING_RATE_START, momentum=0.9, weight_decay=self.WEIGHT_DECAY)
        else:
            raise ValueError("wrong optimizer. only 2 options: \"adam\" or \"sgd\"")
        
        # Learning rate reduced gradually during training
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, 
                                factor=self.LEARNING_RATE_SCHEDULE_FACTOR,
                                patience=self.LEARNING_RATE_SCHEDULE_PATIENCE,
                                mode='max', verbose=True)
        
        
        
        
        
        if self.CUDA == 0:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        # print("y train unique", distinct_label_train, counts_label_train)
        # print("y test unique", distinct_label_test, counts_label_test)
        self.class_weight.to(device)
        # print("class_weight", class_weight)
        
        if self.CLASS_IMBALANCE is None:
            loss_criteria = nn.CrossEntropyLoss()
        elif self.CLASS_IMBALANCE == "weighted_loss":
            loss_criteria = nn.CrossEntropyLoss(weight=self.class_weight)
        elif self.CLASS_IMBALANCE == "oversampling":
            loss_criteria = nn.CrossEntropyLoss()
        else:
            raise ValueError("problem with loss function")
        # Specify the loss criteria
         #
        loss_criteria.to(device)
        
        print("Training on ", device)
        model.to(device)
        # print(model)
        # summary(model, input_size=(1, TARGET_WIDTH, TARGET_HEIGHT, TARGET_DEPTH))
        return model, optimizer, loss_criteria, lr_scheduler, device


    def convert_time(self, start_time, end_time):
        '''
        Convert time (miliseconds) to minutes and seconds
        '''
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    

if __name__ == '__main__':
    pass
    # for X, y in train_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     print("Y: ", y)
    #     break