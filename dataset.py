import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import compress.models.super_pixel


class image(Dataset):
    def __init__(self, root_dir):
        super(image, self).__init__()
        self.root_dir = root_dir
        self.list = os.listdir(root_dir)
        

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        img = cv2.imread(root)
        (h, w, c) = img.shape
        h_ran = np.random.randint(0, (h - 256))
        w_ran = np.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :]/255
        crop = torch.Tensor(crop)
        sample = crop.permute(2, 0, 1)
        return sample

class val_image(Dataset):
    def __init__(self, root_dir):
        super(val_image, self).__init__()
        self.root_dir = root_dir
        self.list = os.listdir(root_dir)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        img = cv2.imread(root)
        crop = img / 255
        crop = torch.Tensor(crop)
        sample = crop.permute(2, 0, 1)
        return sample

class test_image(Dataset):
    def __init__(self, root_dir):
        super(test_image, self).__init__()
        self.root_dir = root_dir
        self.list = os.listdir(root_dir)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        img = cv2.imread(root)
        crop = img / 255
        crop = torch.Tensor(crop)
        sample = crop.permute(2, 0, 1)
        return sample, self.list[idx]


class yuv_image_color(Dataset):
    def __init__(self, root_dir):
        super(yuv_image_color, self).__init__()
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)
        a = len(self.img_list)
        for i in range(a):
            self.img_list[i] = self.img_list[i].strip().replace('\n', '')  # 去掉列表中每一个元素的换行符

    def __len__(self):
        return len(self.img_list)

    def read_img(self, rootdir):
        img = cv2.imread(rootdir)
        (h, w, c) = img.shape
        h_ran = np.random.randint(0, (h - 256))
        w_ran = np.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        yuv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
        yuv_crop = np.transpose(yuv_crop, (2, 0, 1)) / 255

        block_size = 16
        sample_size = 4

        # 创建记录位置的 (1, 256, 256) 数组
        position_map = np.zeros((1, 256, 256))

        # 遍历每个 2x16x16 块
        for i in range(0, yuv_crop.shape[1], block_size):  # 每次处理 16 行
            for j in range(0, yuv_crop.shape[2], block_size):  # 每次处理 16 列
                # 随机选择采样的起始位置
                start_i = np.random.randint(0, block_size - sample_size + 1)
                start_j = np.random.randint(0, block_size - sample_size + 1)

                # 记录采样位置
                position_map[:, i + start_i:i + start_i + sample_size, j + start_j:j + start_j + sample_size] = 1

        # 将原始数组与位置数组相乘
        uv_result = yuv_crop[-2:, :, :] * position_map
        final_result = np.concatenate((yuv_crop[0:1, :, :], uv_result), axis=0)
        return final_result, position_map, yuv_crop

    def __getitem__(self, idx):  # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir + "/" + self.img_list[idx]
        hint_yuv_crop, position_map, ori_yuv_crop = self.read_img(rootdir=root)
        yuv_crop = torch.Tensor(hint_yuv_crop)
        large_array = torch.Tensor(position_map)
        ori_yuv_crop = torch.Tensor(ori_yuv_crop)
        return yuv_crop, large_array, ori_yuv_crop


class yuv_image_color_val(Dataset):
    def __init__(self, root_dir):
        super(yuv_image_color_val, self).__init__()
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)
        a = len(self.img_list)
        for i in range(a):
            self.img_list[i] = self.img_list[i].strip().replace('\n', '')  # 去掉列表中每一个元素的换行符

    def __len__(self):
        return len(self.img_list)

    def read_img(self, rootdir):
        img = cv2.imread(rootdir)
        (h, w, c) = img.shape
        yuv_crop = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv_crop = np.transpose(yuv_crop, (2, 0, 1)) / 255

        block_size = 16
        sample_size = 4

        # 创建记录位置的 (1, 256, 256) 数组
        position_map = np.zeros((1, h, w))

        # 遍历每个 2x16x16 块
        for i in range(0, yuv_crop.shape[1], block_size):  # 每次处理 16 行
            for j in range(0, yuv_crop.shape[2], block_size):  # 每次处理 16 列
                # 随机选择采样的起始位置
                start_i = (block_size - sample_size) // 2
                start_j = (block_size - sample_size) // 2

                # 记录采样位置
                position_map[:, i + start_i:i + start_i + sample_size, j + start_j:j + start_j + sample_size] = 1

        # 将原始数组与位置数组相乘
        uv_result = yuv_crop[-2:, :, :] * position_map
        final_result = np.concatenate((yuv_crop[0:1, :, :], uv_result), axis=0)
        return final_result, position_map, yuv_crop

    def __getitem__(self, idx):  # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir + "/" + self.img_list[idx]
        hint_yuv_crop, position_map, ori_yuv_crop = self.read_img(rootdir=root)
        yuv_crop = torch.Tensor(hint_yuv_crop)
        large_array = torch.Tensor(position_map)
        ori_yuv_crop = torch.Tensor(ori_yuv_crop)
        return yuv_crop, large_array, ori_yuv_crop


class yuv_image_color_spuerpixel(Dataset):
    def __init__(self, root_dir):
        super(yuv_image_color_spuerpixel, self).__init__()
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)
        a = len(self.img_list)
        for i in range(a):
            self.img_list[i] = self.img_list[i].strip().replace('\n', '')  # 去掉列表中每一个元素的换行符

    def __len__(self):
        return len(self.img_list)

    def read_img(self, rootdir):
        img = cv2.imread(rootdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (h, w, c) = img.shape
        h_ran = np.random.randint(0, (h - 256))
        w_ran = np.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        # yuv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
        yuv_crop = np.transpose(crop, (2, 0, 1)) / 255
        return yuv_crop

    def __getitem__(self, idx):  # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir + "/" + self.img_list[idx]
        yuv_crop = self.read_img(rootdir=root)
        yuv_crop = torch.Tensor(yuv_crop)
        return yuv_crop

class yuv_image_color_val_super_pixel(Dataset):
    def __init__(self, root_dir):
        super(yuv_image_color_val_super_pixel, self).__init__()
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)
        self.img_list.sort()
        a = len(self.img_list)
        for i in range(a):
            self.img_list[i] = self.img_list[i].strip().replace('\n', '')  # 去掉列表中每一个元素的换行符

    def __len__(self):
        return len(self.img_list)

    def read_img(self, rootdir):
        img = cv2.imread(rootdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # yuv_crop = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv_crop = np.transpose(img, (2, 0, 1)) / 255
        return yuv_crop

    def __getitem__(self, idx):  # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir + "/" + self.img_list[idx]
        yuv_crop = self.read_img(rootdir=root)
        yuv_crop = torch.Tensor(yuv_crop)
        return yuv_crop, self.img_list[idx]

from compress.models.super_pixel import SuperPixelAttention_3

if __name__ == '__main__':
    data = yuv_image_color_spuerpixel(root_dir="D:/data/coco/val")
    data_loader = DataLoader(dataset=data, batch_size=1)
    spPixel = SuperPixelAttention_3(dims=2)
    spPixel = spPixel.cuda()
    for i, d in enumerate(data_loader):
        d = d.cuda()
        out = spPixel.stoken_forward(d[:, :, :, 1:3])
