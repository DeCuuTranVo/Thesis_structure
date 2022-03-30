from src.predicter import Predicter
import os

# global settings
base_path = '/mnt/data_lab513/vqtran_data'
root_train_unique = os.path.join(base_path, "data", "data_train_dec", "unique")

# GLOBAL VARIABLES
IMAGE_FILENAME = "ADNI_002_S_0295_MR_MP-RAGE__br_raw_20060418193713091_1_S13408_I13722.nii.gz"
IMAGE_LABEL = "CN"

def predict():
    """
    Predict image in sample_dir with trained model

    Args:
        sample_dir (str): Directory of image file to be predicted.

    Returns:
        prediction (dict): Dictionary of propability of 2 classes,
            and predicted class of the image.
    """

    # Create predictor object
    # os.chdir('./src_Tran')
    predictor = Predicter()

    # Predict the image in sample directory
    # os.chdir('..')
    
    image_absolute_path = os.path.join(root_train_unique, IMAGE_FILENAME)
    prediction = predictor.predict(image_absolute_path, IMAGE_LABEL)
    # print(prediction)

    return prediction, predictor.model


def predict(IMAGE_FILENAME, IMAGE_LABEL, have_root=True):
    """
    Predict image in sample_dir with trained model

    Args:
        sample_dir (str): Directory of image file to be predicted.

    Returns:
        prediction (dict): Dictionary of propability of 2 classes,
            and predicted class of the image.
    """

    
        
    # Create predictor object
    # os.chdir('./src_Tran')
    predictor = Predicter()

    # Predict the image in sample directory
    # os.chdir('..')
    
    image_absolute_path = ""
    if not have_root:
        image_absolute_path = os.path.join(root_train_unique, IMAGE_FILENAME)

    else:
        image_absolute_path = IMAGE_FILENAME

        

    prediction = predictor.predict(image_absolute_path, IMAGE_LABEL)
    # print(prediction)

    return prediction, predictor.model


if __name__ == '__main__':
    print(predict("ADNI_002_S_0295_MR_MP-RAGE__br_raw_20060418193713091_1_S13408_I13722.nii.gz", "CN", have_root=False))
