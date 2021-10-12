from src.model import NeuralNetwork
import torch
from torchvision import transforms
import json
from src.trainer import BaselineClassifier
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Predicter():
    '''
    Predict the label of new image base on trained model
    '''

    def __init__(self, model_type='resnet18', using_gpu=True):
        """
        Construct the predicter object.

        Args:
            model_type (str, optional): Type of model architecture.
                Defaults to 'resnet18'.
            using_gpu (bool, optional): GPU enable option. Defaults to True.
        """
        print("PREDICTER CONSTRUCTED")
        # Load training parameters
        params = json.load(open('config/config_predict.json', 'r'))

        # Create CustomTrainer instance with loaded training parameters
        self.trainer = BaselineClassifier(**params)

        # # Check device
        # self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'

        # # Create model
        # self.model = NeuralNetwork(self.trainer.ARCHITECTURE).to(self.device)

        print(self.trainer.PATH_PRETRAINED_MODEL)
        # # Load trained model
        # self.model.load_state_dict(torch.load(os.path.join(
        #     trainer.MODEL_DIR, "trial_" + str(trainer.TRIAL) + ".pth")))

        # Switch model to evaluation mode
        # self.model.eval()
        # Image processing        

    def predict(self):
        """
        Predict image in image_path is peripheral or central.

        Args:
            image_path (str): Directory of image file.

        Returns:
            result (dict): Dictionary of propability of 2 classes,
                and predicted class of the image.
        """
        # Set up DataLoader
        _, _ = self.trainer.set_up_training_data(train_or_val = "val")

        # Set up training params
        model, _, _,_, device = self.trainer.set_up_training()
        # print(device)
        
        # Read image        
        image = self.trainer.x_test[self.trainer.PREDICT_IMAGE_INDEX]
        ground_truth = self.trainer.y_test[self.trainer.PREDICT_IMAGE_INDEX]
        # print("ground_truth", ground_truth)
        
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        
        image = torch.unsqueeze(image, 0)
        
        # print(image.device)
        # print(ground_truth.device)
        # print(image.shape)
        # print(ground_truth.shape)
        # print(ground_truth)
        # Transform image
        # image = self.transform(image)
        # image = image.view(1, *image.size()).to(self.device)
        # Result
        if self.trainer.NUM_CLASSES == 4:
            labels = ['CN', 'EMCI', 'LMCI', 'AD']
            result = {'CN': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3, 'label': -1, 'ground_truth':-1}
        elif self.trainer.NUM_CLASSES == 2:
            if self.trainer.PROBLEM == "CN_AD":
                labels = ['CN', 'AD']
                result = {'CN': 0, 'AD': 1, 'label': -1, 'ground_truth':-1}
            
            elif self.trainer.PROBLEM == "CN_EMCI":
                labels = ['CN', 'EMCI']
                result = {'CN': 0, 'EMCI': 1, 'label': -1, 'ground_truth':-1}
                
            else:
                print("Wrong types of problem")
        else:
            pass
                
        # Predict image
        with torch.no_grad():
            model.eval()
            output = model(image)
            # print(output)
            output_softmax = torch.nn.functional.softmax(output, dim=1)
            # print(output_softmax)
            output_argmax = torch.argmax(output_softmax, dim=1)
            # print("output: ")
            # print(output)
            
            # ps = torch.exp(output)
            # print("ps: ")
            # print(ps)
            # result['prob_central'] = float(ps[0][0].item())
            # result['prob_peripheral'] = float(ps[0][1].item())
            # print(output_argmax)
            
            if self.trainer.NUM_CLASSES == 4:
                result["CN"] = output_softmax[0][0].item()
                result["EMCI"] = output_softmax[0][1].item()
                result["LMCI"] = output_softmax[0][2].item()
                result["AD"] = output_softmax[0][3].item()
                
                if output_argmax == 0:
                    result['label'] = "CN"
                elif output_argmax == 1:
                    result['label'] = "EMCI"
                elif output_argmax == 2:
                    result['label'] = "LMCI"
                elif output_argmax == 3:
                    result['label'] = "AD"
                else:
                    print("Problem!")
                
                result['ground_truth'] = labels[ground_truth]
                
            elif self.trainer.NUM_CLASSES == 2:
                if self.trainer.PROBLEM == "CN_AD":
                    result["CN"] = output_softmax[0][0].item()
                    result["AD"] = output_softmax[0][1].item()
                    
                    if output_argmax == 0:
                        result['label'] = "CN"
                    elif output_argmax == 1:
                        result['label'] = "AD"
                    else:
                        print("Problem!")
                        
                    result['ground_truth'] = labels[ground_truth]
            
                elif self.trainer.PROBLEM == "CN_EMCI":
                    result["CN"] = output_softmax[0][0].item()
                    result["EMCI"] = output_softmax[0][1].item()
                    
                    if output_argmax == 0:
                        result['label'] = "CN"
                    elif output_argmax == 1:
                        result['label'] = "EMCI"
                    else:
                        print("Problem!")
                        
                    result['ground_truth'] = labels[ground_truth]
                
                else:
                    print("Wrong types of problem")
            else:
                pass
                

        return result
