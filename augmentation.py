import nibabel as nib
import matplotlib.pyplot as plt
import torch
import cv2
import torchio as tio
import monai


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
         axes[i].imshow(slice.T, cmap="gray", origin="lower" ,interpolation='none')
         
x_test = torch.load("tensor/x_test_tensor.pt")
y_test = torch.load("tensor/y_test_tensor.pt")
print(x_test.shape)
print(y_test.shape)

img_data_orig = x_test[0]
slice_1 = img_data_orig[img_data_orig.shape[0]//2, :, :]
slice_2 = img_data_orig[:, img_data_orig.shape[1]//2, :]
slice_3 = img_data_orig[:, :, img_data_orig.shape[2]//2]

print(img_data_orig.shape)
print(slice_1.shape)
print(slice_2.shape)
print(slice_3.shape)
# print(torch.max(slice_3))
# print(torch.min(slice_3))
# show_slices([slice_4, slice_5, slice_6])

show_slices([slice_1, slice_2, slice_3])
plt.suptitle("Center slices for augmentation image")  
plt.savefig("origin.png")

transforms = (
     tio.RandomAffine(scales=(0.8,1.2), degrees=(-5,5), translation=(0,0.1), center="image"),
     tio.RescaleIntensity()
)

transform = tio.Compose(transforms)

image_transformation = (
     # monai.transforms.HistogramNormalize(),
     monai.transforms.NormalizeIntensity(),
     monai.transforms.RandStdShiftIntensity(0.9, prob=0.1),
     # monai.transforms.RandAffine(prob=0.9, rotate_range= None, shear_range=None, translate_range=None, scale_range=None), #scale_range=(0.95,1.05) #translate_range=(0,1) #rotate_range=(-0.087*6, 0.087*6)
     monai.transforms.RandFlip(prob=0.5, spatial_axis=None),
     monai.transforms.ToTensor(dtype=None, device=None)
)

image_transformation = monai.transforms.Compose(image_transformation)

img_data_orig = torch.unsqueeze(img_data_orig, 0)
# img_data_transformed = transform(img_data_orig)
img_data_transformed = image_transformation(transform(img_data_orig))
img_data_transformed = torch.squeeze(img_data_transformed)

slice_4 = img_data_transformed[img_data_transformed.shape[0]//2, :, :]
slice_5 = img_data_transformed[:, img_data_transformed.shape[1]//2, :]
slice_6 = img_data_transformed[:, :, img_data_transformed.shape[2]//2]

print(img_data_transformed.shape)
print(slice_4.shape)
print(slice_5.shape)
print(slice_6.shape)

show_slices([slice_4, slice_5, slice_6])
plt.suptitle("Center slices for augmentation image")  
plt.savefig("augmented.png")