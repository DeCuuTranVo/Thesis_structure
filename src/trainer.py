import torch.optim as optim
import torch.nn as nn
import torch
from torchsummary import summary
from src.model import NeuralNetwork
from src.ensemble_model import ResNetEfficientNetEnsemble, ResNetEfficientNetShuffleNetEnsemble, ResNetDenseNetShuffleNetEnsemble
from src.utils import calculate_metrics, create_confusion_matrix, create_classification_report
import os
from src.dataset import CustomImageDataset

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
        self.training_area_under_curve = []
        self.validation_area_under_curve = []
        self.learning_rate_history = []
        self.cross_val_best_metrics = []
        self.cross_test_best_metrics = []
        
        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter(os.path.join(self.LOG_DIR, "trial_" + str(self.TRIAL)))  #'runs/resnet_adam_110'
        # print("tensorboard recording at: ",os.path.join(self.LOG_DIR, "trial_" + str(self.TRIAL)))
        
        self.saved_model_path = os.path.join(self.MODEL_DIR,"trial_" + str(self.TRIAL) + ".pth")
        self.x_cross_val = 0
        self.y_cross_val = 0
        self.x_train_val = 0
        self.y_train_val = 0
        self.x_train = 0
        self.x_test = 0
        self.y_train = 0
        self.y_test = 0
        self.class_weight = []
        # print(type(self.class_weight))
        
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
            
            # print("output_shape:", output.shape)
            # print("output:",output)
            
            # target = torch.nn.functional.one_hot(target)
            # print("target_shape:", target.shape)
            # print("target",target)
            
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
        
        accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score = calculate_metrics(out_gt, out_pred, self.NUM_CLASSES)
        # with open("output.txt", "a") as f:        
        #     print("Epoch:", epoch, file =f)
        #     print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
        #     train_loss / (batch + 1), 100 * accuracy, auc_score), file =f)

        print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            avg_loss, 100 * accuracy, auc_score))
        
        
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
            avg_loss, 100 * accuracy, auc_score), file=f)

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
            out_gt, out_pred, self.NUM_CLASSES)
        
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
            self.writer.add_scalar("Loss/val", avg_loss,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Accuracy/val", accuracy,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Precision/val", precision,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Recall/val", recall,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("F1_score/val", f1_score,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Sensitivity/val", sensitivity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Specificity/val", specificity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("AUROC/val", auc_score,  epoch+self.START_RECORD_EPOCH)
            
            with open(self.OUTPUT_FILE, 'a') as f:
                print('Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
                    avg_loss, 
                    100. * accuracy,
                    auc_score), file=f)
        elif train_or_validation == "validation":
            # print("line 194",len(out_gt))
            con_mat = create_confusion_matrix(out_gt, out_pred, self.PROBLEM)
            print("confusion matrix")
            print(con_mat)
            
            class_report = create_classification_report(out_gt, out_pred, self.PROBLEM)
            print("classification report")
            print(class_report)
            
            self.writer.add_scalar("Loss/test", avg_loss,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Accuracy/test", accuracy,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Precision/test", precision,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Recall/test", recall,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("F1_score/test", f1_score,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Sensitivity/test", sensitivity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("Specificity/test", specificity,  epoch+self.START_RECORD_EPOCH)
            self.writer.add_scalar("AUROC/test", auc_score,  epoch+self.START_RECORD_EPOCH)
            
            
        else:
            print("Invalid training or validation")
        # return average loss for the epoch
        return avg_loss, accuracy, auc_score #correct / len(test_loader.dataset) # accuracy


    def set_up_training_data(self, train_or_val = "train"):
        '''
        Return: 
            train_dataloader, val_dataloader
        '''
        # print('----- Setting up data ... -----')
        self.x_train = torch.load(self.X_TRAIN_PATH)
        self.x_test = torch.load(self.X_TEST_PATH)
        self.y_train = torch.load(self.Y_TRAIN_PATH)
        self.y_test = torch.load(self.Y_TEST_PATH)

        # print(self.x_train.shape)
        # print(self.x_test.shape)
        # print(self.y_train.shape)
        # print(self.y_test.shape)

        # print("load complete")

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
            train_or_val="train",
            rotation=self.ROTATE,
            translation=self.TRANSLATE,
            scaling=self.SCALE)  # augment = True
        test_dataset = CustomImageDataset(
            self.x_test,
            self.y_test,
            train_or_val = "val",
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
        # print("self.class_weight 307:",self.class_weight)
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
                # print("Using oversampling")
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
                # print("using weighted_loss")
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
             
        # print("number of sample in train dataloader",len(train_dataloader.dataset))
        # print("number of sample in test dataloader",len(test_dataloader.dataset))
        # print('Complete DataLoader')
        return train_dataloader, test_dataloader

    def set_up_training(self):
        '''
        In json setting: architecture should be in order: resnet > efficientnet > shufflenet > densenet > mobilenet
        
        Return:
            model, optimizer, criterion, scheduer, device
        '''
        model = None
        if len(self.ARCHITECTURE.keys()) == 0:
            raise ValueError("no available model")
        elif len(self.ARCHITECTURE.keys()) == 1:
            model_kind = list(self.ARCHITECTURE.keys())[0]
            model = NeuralNetwork(self.ARCHITECTURE[model_kind], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)     
        elif len(self.ARCHITECTURE.keys()) == 2:
            model_resnet = NeuralNetwork(self.ARCHITECTURE["RESNET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
            model_efficientnet = NeuralNetwork(self.ARCHITECTURE["EFFICIENTNET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)        
            model = ResNetEfficientNetEnsemble(model_resnet, model_efficientnet, nb_classes = self.NUM_CLASSES)
        elif len(self.ARCHITECTURE.keys()) == 3:
            if "RESNET" in list(self.ARCHITECTURE.keys()) and "EFFICIENTNET" in list(self.ARCHITECTURE.keys()) and "SHUFFLENET" in list(self.ARCHITECTURE.keys()):
                model_resnet = NeuralNetwork(self.ARCHITECTURE["RESNET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
                model_efficientnet = NeuralNetwork(self.ARCHITECTURE["EFFICIENTNET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)        
                model_shufflenet = NeuralNetwork(self.ARCHITECTURE["SHUFFLENET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
                model = ResNetEfficientNetShuffleNetEnsemble(model_resnet, model_efficientnet, model_shufflenet, nb_classes = self.NUM_CLASSES)
            elif "RESNET" in list(self.ARCHITECTURE.keys()) and "DENSENET" in list(self.ARCHITECTURE.keys()) and "SHUFFLENET" in list(self.ARCHITECTURE.keys()):
                model_resnet = NeuralNetwork(self.ARCHITECTURE["RESNET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
                model_densenet = NeuralNetwork(self.ARCHITECTURE["DENSENET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)        
                model_shufflenet = NeuralNetwork(self.ARCHITECTURE["SHUFFLENET"], num_classes=self.NUM_CLASSES, is_pretrained=True, dropout_rate=self.DROPOUT_RATE)
                model = ResNetDenseNetShuffleNetEnsemble(model_resnet, model_densenet, model_shufflenet, nb_classes = self.NUM_CLASSES)
        else:          
            raise ValueError("four architecture combination is not available")
        
        # print("model", model)
        
        
        if self.USE_TRAINED_MODEL:
            if self.PATH_PRETRAINED_MODEL is None:
                if os.path.exists(self.saved_model_path):            
                    raise ValueError("This part is gonna be modified with state dict loading keywords errors")
                    # model.load_state_dict(torch.load(self.saved_model_path))
                    # print("load trained model from", self.saved_model_path)
                else:
                    print("trained model",self.saved_model_path ,"not exist!")
                    print("training from scratch")
            else:
                model_dict = model.state_dict()
                # print("Initial model:",model.state_dict().keys())
                # print("Initial model:",model.state_dict()['model_resnet.base_model.conv1.weight'].shape)
                # print("Initial model:",model.state_dict()['model_resnet.base_model.conv1.weight'].mean() )
                # print("Initial model:",model.state_dict()['model_resnet.base_model.conv1.weight'].std() )
                pretrained_dict =  torch.load(self.PATH_PRETRAINED_MODEL)
                update_dict = {}
                
                # print(model)
                # print("Keys of pretrained dict:", list(pretrained_dict.keys())[:10])
                # print("Pretrained dict before compare:", pretrained_dict["module.model_resnet.base_model.conv1.weight"].shape)
                
                
                model_dict_key = list(model_dict.keys())
                # print("model_dict_key: ",model_dict_key[:10])
                for k, v in pretrained_dict.items():
                    k_replace = k.replace('module.', '') #???
                    
                    
                    # print("k_replace:",k_replace)
                    # print("model_dict_key",model_dict_key)
                    have_key_words = (k_replace in model_dict_key) #first conditions: same keywords
                    # print(have_key_words)
                    
                    # print("k: ", k)
                    # print(model_dict[k_replace].shape)
                    # print(pretrained_dict[k].shape)
                    weight_same_shape = (model_dict[k_replace].shape == pretrained_dict[k].shape) # second conditions: weight same shape
                    # print(weight_same_shape)
                    
                    
                    if have_key_words and weight_same_shape:
                        # print(k)
                        # print(model_dict[k_replace])
                        # print(model_dict[k_replace].shape)
                        
                        # print(pretrained_dict[k])
                        # print(pretrained_dict[k].shape)
                        
                        update_dict[k_replace] = v
                        # print(update_dict)
                        
                        
                
                ## pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                ##       (k in model_dict) and (model_dict[k].shape == pretrained_dict["module." + k].shape)}
                
                # print("Update dict after compare:", update_dict["model_resnet.base_model.conv1.weight"].shape)
                model_dict.update(update_dict)
                model.load_state_dict(model_dict)
                # print("After load pretrained weight:",model.state_dict().keys())
                # print("After load pretrained weight:",model.state_dict()['model_resnet.base_model.conv1.weight'].shape)
                # print("After load pretrained weight:",model.state_dict()['model_resnet.base_model.conv1.weight'].mean())
                # print("After load pretrained weight:",model.state_dict()['model_resnet.base_model.conv1.weight'].std())
                del model_dict, pretrained_dict, update_dict
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
        if self.SCHEDULER == "reduce":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, 
                                factor=self.LEARNING_RATE_SCHEDULE_FACTOR,
                                patience=self.LEARNING_RATE_SCHEDULE_PATIENCE,
                                mode='max', verbose=True)
        elif self.SCHEDULER == "cyclic":       
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                                optimizer, 
                                base_lr=self.LEARNING_RATE_START, 
                                max_lr=self.MAX_LEARING_RATE, 
                                step_size_up=self.STEP_SIZE_UP, 
                                cycle_momentum=False, 
                                mode=self.MODE,
                                gamma=self.GAMMA,
                                verbose=True)
        else:
            raise ValueError("wrong argument for scheduler. Only 2 options: reduce or cyclic")
        
        # print("scheduler: ",lr_scheduler)
        
        
        if self.CUDA == 0:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self.CUDA == 1:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        elif self.CUDA == 2:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        elif self.CUDA == -1:
            device = torch.device("cpu")
        else:
            raise ValueError("Wrong argument: device only have 4 values: 0, 1, 2, and -1")
        

        try:
            self.class_weight.to(device)
        except:
            pass
        # print("class_weight", class_weight)
        
        # Specify the loss criteria
        if self.CLASS_IMBALANCE is None:
            loss_criteria = nn.CrossEntropyLoss()
        elif self.CLASS_IMBALANCE == "weighted_loss":
            # print("self.class_weight 491:", self.class_weight)
            loss_criteria = nn.CrossEntropyLoss(weight=self.class_weight)
        else:
            raise ValueError("problem with loss function")
        
        loss_criteria.to(device)
        
        # print("Training on ", device)
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
    
    
    def load_training_data_cross_validation(self):
        # print(type(self.class_weight))
        '''
        Return: 
            train_dataloader, val_dataloader
            
        Dont call this function at the same time with set_up_training_data
        '''
        # print('----- Setting up data for cross validation... -----')
        self.x_cross_val = torch.load(self.X_CV_PATH)
        self.y_cross_val = torch.load(self.Y_CV_PATH)
        
        # print(self.x_cross_val.shape)
        # print(self.y_cross_val.shape)
        
        self.x_cross_val = torch.unsqueeze(self.x_cross_val, 1)
        self.y_cross_val = self.y_cross_val.type(torch.long) #long  
        
        distinct_label_cv, counts_label_cv = torch.unique(self.y_cross_val, return_counts=True)

        # num_label_total = len(self.y_cross_val) 
        # print(num_label_total)

        # print(type(self.class_weight))
        for label in distinct_label_cv.tolist():
            self.class_weight.append(1.0/(counts_label_cv[label])) #float(num_label_total)
        
        self.class_weight = torch.FloatTensor(self.class_weight)
        # print("class_weight: ",self.class_weight)     
        
        
    def set_up_training_data_cross_validation(self, train_ids, val_ids, fold, outer_inner="outer"):
        try:
            os.mkdir(os.path.join(self.MODEL_DIR,"trial_" + str(self.TRIAL)))
        except:
            pass
        
        
        
        self.writer = SummaryWriter(os.path.join(self.LOG_DIR, "trial_" + str(self.TRIAL), "fold_" + str(fold))) 
        self.saved_model_path = os.path.join(self.MODEL_DIR,"trial_" + str(self.TRIAL),"fold_" + str(fold) + ".pth")
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        if outer_inner == "outer":
            dataset_train = CustomImageDataset(
                self.x_cross_val,
                self.y_cross_val,
                train_or_val = "train",
                augment=True)
            
            dataset_val = CustomImageDataset(
                self.x_cross_val,
                self.y_cross_val,
                train_or_val = "val",
                augment=False)          
        elif outer_inner == "inner":
            dataset_train = CustomImageDataset(
                self.x_train_val,
                self.y_train_val,
                train_or_val = "train",
                augment=True)
            
            dataset_val = CustomImageDataset(
                self.x_train_val,
                self.y_train_val,
                train_or_val = "val",
                augment=False)
        else: 
            raise ValueError("Wrong outer_inner arguments: only chose outer or inner")
           
        train_dataloader = torch.utils.data.DataLoader(dataset_train, 
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False,
            sampler = train_subsampler
        ) # create your dataloader
        val_dataloader = torch.utils.data.DataLoader(dataset_val, 
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False,
            sampler = val_subsampler) #torch.utils.data.sampler.WeightedRandomSampler(weight_sequence_test, int(counts_label_test[0]*4))) #None
             
        # print("number of sample in train dataloader",len(train_dataloader.dataset))
        # print("number of sample in val dataloader",len(val_dataloader.dataset))
        # print('Complete DataLoader')
        return train_dataloader, val_dataloader

if __name__ == '__main__':
    pass
    # for X, y in train_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     print("Y: ", y)
    #     break