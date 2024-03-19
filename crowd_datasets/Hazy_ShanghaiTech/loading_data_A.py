import torchvision.transforms as standard_transforms
# from .SHHA import SHHA # train.py
from .SHHA_A import SHHA # train4.py

# add by wedream
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
from torchvision import utils as vutils
 
def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # print ("len_input_tensor: ",len(input_tensor.shape),input_tensor.shape[0] )
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)
# end

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(data_root):
    # the pre-proccssing transform
    # 不做正态分布，直接归一化，便于复原模块
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        # 标准化处理-->转换为标准正太分布，使模型更容易收敛
        # standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA(data_root, train=False, transform=transform)

    return train_set, val_set
