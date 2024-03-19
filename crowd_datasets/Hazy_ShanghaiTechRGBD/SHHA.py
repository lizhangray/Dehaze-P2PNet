import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        # self.train_lists = "shanghai_tech_part_a_train.list"
        # self.eval_list = "shanghai_tech_part_a_test.list"
        self.train_lists = "crowd_datasets/Hazy_ShanghaiTechRGBD/train.list"
        self.eval_list = "crowd_datasets/Hazy_ShanghaiTechRGBD/test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        # add by wedream
        hr_path = img_path.replace('images','HR')
        # print("hr_path:{}".format(hr_path))
        # end
        gt_path = self.img_map[img_path]
        # load image and ground truth
        # print("img_path:{}".format(img_path))
        # print("gt_path:{}".format(gt_path))
        # img, point = load_data((img_path, gt_path), self.train)
        img, hr, point = load_data((img_path, hr_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)
            # add by wedream
            hr = self.transform(hr)
            # end

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                hr = torch.nn.functional.upsample_bilinear(hr.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            img, hr, point = random_crop(img, hr, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            hr = torch.Tensor(hr[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # print("img:{}".format(img))
        hr = torch.Tensor(hr)

        # add by wedream
        # from .loading_data import DeNormalize,save_image_tensor
        # # 反归一化，恢复成原图像
        # denorm = DeNormalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        # img_denorm = denorm(img)
        # hr_denorm = denorm(hr)
        # print("img_denorm[0]:{}".format(img_denorm[0]))
        # print("hr_denorm[0]:{}".format(hr_denorm[0]))
        # save_image_tensor(img_denorm[0].unsqueeze(0), 'img_denorm0.png')
        # save_image_tensor(hr_denorm[0].unsqueeze(0), 'hr_denorm0.png')
        # exit(0)
        # end

        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        # return img, target
        return img, hr, target


def load_data(img_gt_path, train):
    # img_path, gt_path = img_gt_path
    img_path, hr_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_hr = cv2.imread(hr_path)
    img_hr = Image.fromarray(cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    # return img, np.array(points)
    return img, img_hr, np.array(points)

# random crop augumentation
def random_crop(img, hr, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_hr = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        result_hr[i] = hr[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_hr, result_den
