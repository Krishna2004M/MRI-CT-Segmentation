import torch
from monai.networks.nets import UNet

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=1,
).cuda()

# (Perform training using your training dataset)
# ...

# Once training is done, save the model
torch.save(model.state_dict(), 'segmentation_model.pth')
