from src.model import NeuralNetwork
import torch
from torchvision import transforms
import json
from src.trainer import BaselineClassifier
from src.utils import seed_torch
import nibabel as nib
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchio as tio

# global settings
base_path = '/mnt/data_lab513/vqtran_data'
root_data = os.path.join(base_path, "data", "raw_data", "ADNI_NIfTI")
root_bias_correction = os.path.join(base_path, "data", "clean_data", "mri_bias_correction")
root_bet = os.path.join(base_path, "data", "clean_data", "mri_brain_extraction")
root_reg = os.path.join(base_path, "data", "clean_data", "mri_registration")
root_meta = os.path.join(base_path, "data", "meta_data")#, "Pre-Thesis_metadata", "ADNI") 
root_train = os.path.join(base_path, "data", "train_data")
root_train_dec = os.path.join(base_path, "data", "data_train_dec", "origin")
root_train_unique = os.path.join(base_path, "data", "data_train_dec", "unique")


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
        # print("PREDICTER CONSTRUCTED")
        # Load training parameters
        params = json.load(open('config/config_predict.json', 'r'))

        # Create CustomTrainer instance with loaded training parameters
        self.trainer = BaselineClassifier(**params)

        # set random state
        seed_torch(seed=self.trainer.SEED)
        
        # # Check device
        # self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'

        # # Create model
        # self.model = NeuralNetwork(self.trainer.ARCHITECTURE).to(self.device)

        # print(self.trainer.PATH_PRETRAINED_MODEL)
        # # Load trained model
        # self.model.load_state_dict(torch.load(os.path.join(
        #     trainer.MODEL_DIR, "trial_" + str(trainer.TRIAL) + ".pth")))

        # Switch model to evaluation mode
        # self.model.eval()
        # Image processing        
        
        # Set up DataLoader
        _, _ = self.trainer.set_up_training_data(train_or_val = "val")

        # Set up training params
        self.model, _, _,_, self.device = self.trainer.set_up_training()
        # print(device)

    def predict(self, image_absolute_path, ground_truth): # need to convert ground truth to number
        """
        Predict image in image_path is peripheral or central.

        Args:
            image_path (str): Directory of image file.

        Returns:
            result (dict): Dictionary of propability of 2 classes,
                and predicted class of the image.
        """

        
        # Read image        
        
        # image_absolute_path = os.path.join(root_train_unique,"ADNI_002_S_0295_MR_MP-RAGE__br_raw_20060418193713091_1_S13408_I13722.nii.gz")
        sample_img = nib.load(image_absolute_path)
        image = sample_img.get_fdata() #get image in numpy format
        
        # print(image.shape) #(182, 218, 182)
        # print(image.dtype) #float64
        
        image_tensor = torch.Tensor(image)
        image_tensor = torch.unsqueeze(image_tensor,0)
        # print(image_tensor.shape) #torch.Size([182, 218, 182])
        # print(image_tensor.dtype) #torch.float32
        
        image_transformation_tio_predict = tio.transforms.Compose(
            [
            tio.transforms.Resize((110,110,110)),
            tio.transforms.ZNormalization(),
            tio.RescaleIntensity(),
            ]
        )
        
        image = image_transformation_tio_predict(image_tensor)
        image = torch.unsqueeze(image,0)
        # print(image.shape) #torch.Size([182, 218, 182])
        # print(image.dtype) #torch.float32      
        # print(torch.mean(image))
        # print(torch.max(image))
        # print(torch.min(image))
        # print(torch.std(image))
        
            # image = self.trainer.x_test[self.trainer.PREDICT_IMAGE_INDEX]
            # ground_truth = self.trainer.y_test[self.trainer.PREDICT_IMAGE_INDEX]`

        # print("ground_truth", ground_truth)        
        
        # Result
        if self.trainer.NUM_CLASSES == 3:
            labels = ['CN', 'EMCI', 'AD']
            result = {'CN': 0, 'EMCI': 1, 'AD': 2, 'label': -1, 'ground_truth':-1}
            ground_truth = labels.index(ground_truth)   
        elif self.trainer.NUM_CLASSES == 4:
            labels = ['CN', 'EMCI', 'LMCI', 'AD']
            result = {'CN': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3, 'label': -1, 'ground_truth':-1}
            ground_truth = labels.index(ground_truth)
        elif self.trainer.NUM_CLASSES == 2:
            if self.trainer.PROBLEM == "CN_AD":
                labels = ['CN', 'AD']
                result = {'CN': 0, 'AD': 1, 'label': -1, 'ground_truth':-1}
                ground_truth = labels.index(ground_truth)
                # print("ground_truth 128:", ground_truth)
            
            elif self.trainer.PROBLEM == "CN_EMCI":
                labels = ['CN', 'EMCI']
                result = {'CN': 0, 'EMCI': 1, 'label': -1, 'ground_truth':-1}
                ground_truth = labels.index(ground_truth)
                
            else:
                raise ValueError("Problem with problem types")
        else:
            raise ValueError("Probelm with number of classes")
        # print("ground_truth 139: ",ground_truth)
        image = image.to(self.device)
        # ground_truth = torch.Tensor([ground_truth])
        # print("ground_truth 142: ",ground_truth)
        # ground_truth = ground_truth.to(self.device) #????  
        # print("ground_truth 144: ",ground_truth)
        
        # print(image.device)
        # print(ground_truth.device)
        # print(image.shape)
        # print(ground_truth.shape)
        # print(ground_truth)
        # Transform image
        # image = self.transform(image)
        # image = image.view(1, *image.size()).to(self.device)
                
        # Predict image
        with torch.no_grad():
            self.model.eval()
            output = self.model(image)
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
            
            
            # ground_truth = int(ground_truth.item())
            if self.trainer.NUM_CLASSES == 3:
                result["CN"] = output_softmax[0][0].item()
                result["EMCI"] = output_softmax[0][1].item()
                result["AD"] = output_softmax[0][2].item()
                
                if output_argmax == 0:
                    result['label'] = "CN"
                elif output_argmax == 1:
                    result['label'] = "EMCI"
                elif output_argmax == 2:
                    result['label'] = "AD"
                else:
                    raise ValueError("Problem!")
                
                result['ground_truth'] = labels[ground_truth]
            elif self.trainer.NUM_CLASSES == 4:
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
                    raise ValueError("Problem!")
                
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
                        raise ValueError("Problem!")
                        
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
                    raise ValueError("Wrong types of problem")
            else:
                raise ValueError("Wrong numclass setting: only 2 and 4 available")
                

        return result
