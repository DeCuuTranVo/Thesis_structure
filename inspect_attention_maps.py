import nibabel as nib
import matplotlib.pyplot as plt
import torch
import cv2

# sample_dir = os.path.join(base_dir,"data","raw_data",'ADNI_NIfTI', 'ADNI_002_S_0295_MR_MP-RAGE__br_raw_20060418193713091_1_S13408_I13722.nii')
sample_dir = "/home/vqtran/Thesis/attention_maps/layer1.1.conv2/attention_map_0_0_0.nii.gz"
sample_img = nib.load(sample_dir)
sample_img_data = sample_img.get_fdata()
print(sample_img_data.shape)


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
         axes[i].imshow(slice.T, cmap="gray", origin="lower")
         
def show_slices_gradcam(slices_orig, slices_gradcam):
    """ Function to display row of image slices """
    assert len(slices_orig) == len(slices_gradcam)

    fig, axes = plt.subplots(1, len(slices_orig))
    for i in range(len(slices_orig)):
         axes[i].imshow(slices_orig[i].T, cmap="gray", origin="lower",interpolation='none')
         axes[i].imshow(slices_gradcam[i].T, cmap="jet", origin="lower", interpolation='none',alpha=0.5)
            
#img_data = np.squeeze(sample_img_data,3)
img_data = sample_img_data
slice_0 = img_data[img_data.shape[0]//2, :, :]
slice_1 = img_data[:, img_data.shape[1]//2, :]
slice_2 = img_data[:, :, img_data.shape[2]//2]

slice_0 = cv2.resize(slice_0, (110, 110),interpolation = cv2.INTER_NEAREST)
slice_1 = cv2.resize(slice_1, (110, 110),interpolation = cv2.INTER_NEAREST)
slice_2 = cv2.resize(slice_2, (110, 110),interpolation = cv2.INTER_NEAREST)
print(slice_0.shape)
# show_slices([slice_0, slice_1, slice_2])

#####################################################
# import original image
x_test = torch.load("tensor/x_test_tensor.pt")
y_test = torch.load("tensor/y_test_tensor.pt")
print(x_test.shape)
print(y_test.shape)

img_data_orig = x_test[0]
slice_4 = img_data_orig[img_data_orig.shape[0]//2, :, :]
slice_5 = img_data_orig[:, img_data_orig.shape[1]//2, :]
slice_6 = img_data_orig[:, :, img_data_orig.shape[2]//2]

# #np_arr = torch_tensor.cpu().detach().numpy()
# slice_4 = cv2.resize(slice_4.numpy(), (110, 110),interpolation = cv2.INTER_NEAREST)
# slice_5 = cv2.resize(slice_5.numpy() , (110, 110),interpolation = cv2.INTER_NEAREST)
# slice_6 = cv2.resize(slice_6.numpy(), (110, 110),interpolation = cv2.INTER_NEAREST)
print(slice_4.shape)
print(slice_5.shape)
print(slice_6.shape)

# print(torch.max(slice_4))
# print(torch.min(slice_4))
# show_slices([slice_4, slice_5, slice_6])

show_slices_gradcam([slice_4, slice_5, slice_6], [slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  
plt.savefig("GradCAM.png")

show_slices([slice_4, slice_5, slice_6])
plt.savefig("show_slices_test.png")
# import skimage
# import matplotlib.pyplot as plt
# #from skimage.util.montage2d import montage2d
# from skimage.util import montage as montage2d
# fig, ax1 = plt.subplots(1, 1, figsize = (10, 20))
# ax1.imshow(montage2d(np.squeeze(sample_img_data,3).T), cmap ='bone')
# #fig.savefig('ct_scan.png')

