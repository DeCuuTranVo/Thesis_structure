from src.predicter import Predicter
import os
import json
import pandas as pd

# global settings
base_path = '/mnt/data_lab513/vqtran_data'
root_train_unique = os.path.join(base_path, "data", "data_train_dec", "unique")

#THIS VERSION IS ONLY FOR 2 CLASSES
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
    
    # Load subjects dict for model_evaluation and dataset_evaluation
    subject_dict = json.load(open('investigate/unique_dataset_dict.json', 'r'))
    # print(subject_list)
    
    subject_dict_with_prediction = {}
    
    for key in subject_dict.keys():
        
        global prediction
        prediction = 0
        # for 2 class NC-AD problem # need to change for generalization
        if (subject_dict[key][1] == "CN") or (subject_dict[key][1] == "EMCI") or (subject_dict[key][1] == "AD"):
            # print(key)
            image_absolute_path = os.path.join(root_train_unique, subject_dict[key][2])
            # print(image_absolute_path)
            prediction = predictor.predict(image_absolute_path, subject_dict[key][1])
            subject_dict_with_prediction[key] = {"Subject ID": key,
                                                 "Image ID": subject_dict[key][0],
                                                 "Image Path": image_absolute_path, 
                                                 "Image Target": subject_dict[key][1], 
                                                **prediction}
            
            # print(subject_dict_with_prediction)
            # print(prediction)
            
    print(len(subject_dict_with_prediction.keys()))
    # print(prediction)

    # Serializing json
    subject_with_prediction_unique_json_object = json.dumps(subject_dict_with_prediction, indent = 4)

    # Writing to sample.json
    with open("investigate/unique_subject_with_prediction_dict.json", "w") as outfile:
        outfile.write(subject_with_prediction_unique_json_object)
    
    df = pd.DataFrame()
    unique_subject_with_prediction_dict = json.load(open('investigate/unique_subject_with_prediction_dict.json', 'r'))

    for key in unique_subject_with_prediction_dict.keys():
        # print(unique_subject_with_prediction_dict[key])     
        df = df.append(unique_subject_with_prediction_dict[key], ignore_index = True)

        
    df.to_csv("./investigate/unique_subject_with_prediction.csv")
    # print(df.head())
    return 0 #prediction


if __name__ == '__main__':
    print(predict())
