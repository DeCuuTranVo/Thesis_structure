# Thesis structure

Alzheimer’s Disease (AD) is one of the temporarily leading causes of death for the elderly in the United States. AD patients undergo tremendous cognitive function degradation, entailing memory loss, executive function damage, language impairment, and visuospatial dysfunction, which negatively affect their life quality and life expectancy. Currently, there are no available cures for AD. Thus, early diagnosis plays an integral role in decelerating the disease’s progression and improving patients’ symptoms. Among AD diagnosis methods, magnetic resonance imaging (MRI) has advantages in non-invasiveness, fast procedure, and availability with acceptable accuracy. A computer-aided diagnosis (CAD) system empowered by deep learning algorithms has the potential for MRI-based automatic detection of AD onset and classification of AD progress. 

In this work, we develop an automatic classification pipeline for classification between three groups of patients: normal control (NC), early mild cognitive impairment (EMCI), and AD. The pipeline comprises of four stages: (1) Preprocessing raw structural MRI images from ADNI dataset (N3 bias correction, brain extraction, and brain registration), (2) Training customized versions of 3D-ResNet, 3D-EfficientNet, and 3D-ShuffleNet, (3) Filtering dataset for descriptive and distinctive images (4) Combining the abovementioned architectures and test performances on the filtered dataset. Out method uses affine augmentation, dropout, and weight decay to combat overfitting. Furthermore, hyperparameter tuning is implemented in a grid search fashion to facilitate convergence. With a stratified nested 5-fold cross-validation, our whole-brain three-dimensional ensembled deep learning approach achieved an accuracy of 96% on three-way classification (AD versus EMCI versus CN), 97% on NC-AD binary classification, and 100% on NC-EMCI binary classification. In addition, gradient-weighted class activation mapping (Grad-CAM) is implemented to visualize the decision-making process of the trained models. This study validates the application of ensembled deep learning algorithms in medical image analysis and develops a CAD system for AD screening and assessment. 

Model architecture

![Alt text](./ensemble-model-architecture.png?raw=true "Title")

Model prediction example

![Alt text](./Model-prediction.png?raw=true "Title")
