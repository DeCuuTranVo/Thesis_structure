import monai



######################## 3D DenseNet121
# keyword_arguments = {'spatial_dims': 3,
#                     'in_channels': 3,
#                     'out_channels': 4}
# my_model = monai.networks.nets.DenseNet121(init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), pretrained=False, progress=True, **keyword_arguments)
# print(my_model)

######################## 3D EfficientNet Batchnorm
# my_model =  monai.networks.nets.EfficientNetBN("efficientnet-b0", pretrained=True, progress=True, spatial_dims=3, in_channels=3, num_classes=4, norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)
# print(my_model)

######################### 3D SENet154 
# keyword_arguments = {'spatial_dims': 3,
#                     'in_channels': 3,
#                     "num_classes":4}
# my_model =  monai.networks.nets.SENet154(layers=(3, 8, 36, 3), groups=64, reduction=16, pretrained=False, progress=True, **keyword_arguments)
# print(my_model)

######################### 3D SEResNet50
# keyword_arguments = {'spatial_dims': 3,
#                     'in_channels': 3,
#                     "num_classes":4}
# my_model = monai.networks.nets.SEResNet50(layers=(3, 4, 6, 3), groups=1, reduction=16, dropout_prob=None, inplanes=64, downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **keyword_arguments)
# print(my_model)

######################### 3D SEResNext50
keyword_arguments = {'spatial_dims': 3,
                    'in_channels': 3,
                    "num_classes":4}
my_model =  monai.networks.nets.SEResNext50(layers=(3, 4, 6, 3), groups=32, reduction=16, dropout_prob=None, inplanes=64, downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **keyword_arguments)
print(my_model)

####################### 3D Variational Autoencoder #################
my_model = monai.networks.nets.VarAutoEncoder(spatial_dims=3,
    in_shape=(32, 32),  # image spatial shape
    out_channels=4,
    latent_size=4,
    channels=(16, 32, 64),
    strides=(1, 2, 2), kernel_size=3, up_kernel_size=3, num_res_units=0, inter_channels=None, inter_dilations=None, num_inter_units=2, act='PRELU', norm='INSTANCE', dropout=None, bias=True)
print(my_model)