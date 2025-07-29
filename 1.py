import os
import cv2
from concurrent.futures import ThreadPoolExecutor

import torch

# from dataset import yuv_image_color

# 定义处理图像的函数
def process_image(img_name, root):
    img_dir = os.path.join(root, img_name)
    img = cv2.imread(img_dir)
    if img is not None:
        h, w, c = img.shape
        if h <= 256 or w <= 256:
            os.remove(img_dir)
            print(f"Deleted image: {img_name}")

# 根目录
root = "/data2/zgy/data/coco/train2017"
img_list = [img.strip().replace('\n', '') for img in os.listdir(root)]

# 使用线程池，指定最大线程数为8
with ThreadPoolExecutor(max_workers=8) as executor:
    # 提交任务到线程池，从第40000张图片开始
    for img_name in img_list:
        executor.submit(process_image, img_name, root)


def rgb_to_yuv(rgb_tensor):
    # 获取批量大小、高度和宽度
    b, _, h, w = rgb_tensor.shape

    # 将RGB分量分离
    r = rgb_tensor[:, 0, :, :]
    g = rgb_tensor[:, 1, :, :]
    b = rgb_tensor[:, 2, :, :]

    # 计算Y分量
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # 计算U分量
    u = -0.14713 * r - 0.28886 * g + 0.436 * b

    # 计算V分量
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    # 将Y、U、V分量合并成一个Tensor
    yuv_tensor = torch.stack([y, u, v], dim=1)

    return yuv_tensor


def yuv_to_rgb(yuv_tensor):
    # 首先将YUV值从0 - 255归一化到0 - 1
    yuv_tensor = yuv_tensor / 255.0
    
    # 分离YUV分量
    y = yuv_tensor[:, 0, :, :]
    u = yuv_tensor[:, 1, :, :]
    v = yuv_tensor[:, 2, :, :]
    
    # 根据归一化后的值进行转换
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    
    # 将R、G、B分量合并
    rgb_tensor = torch.stack([r, g, b], dim=1)
    
    # 将RGB值从0 - 1反归一化回0 - 255，并确保值在0 - 255范围内
    rgb_tensor = torch.clamp(rgb_tensor * 255.0, 0, 255)
    
    # 如果需要，可以将浮点数转换为整数
    rgb_tensor = rgb_tensor.to(torch.uint8)
    
    return rgb_tensor

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset import image, val_image

def save_tensor_as_image(rgb_tensor, file_path):
    """
    将 RGB 张量存储为图片
    :param rgb_tensor: 形状为 (b, c, h, w) 的 RGB 张量，值范围为 [0, 255]
    :param file_path: 保存图片的路径（包括文件名和扩展名）
    """
    # 检查输入形状
    assert rgb_tensor.ndim == 4 and rgb_tensor.shape[1] == 3, "RGB tensor must have shape (b, 3, h, w)"
    
    # 取第一个 batch 的图片
    image_tensor = rgb_tensor[0]  # 形状: (3, h, w)
    
    # 转换维度为 (h, w, c) 并转换为 NumPy 数组
    image_np = image_tensor.permute(1, 2, 0).byte().cpu().numpy()
    
    # 转换为 Pillow 图片
    image = Image.fromarray(image_np, mode="RGB")
    
    # 保存图片
    image.save(file_path)

def save_tensor_as_yuv(tensor, filename, format='YUV444p'):
    # 确保Tensor的值域为0 - 255，并且是整数类型
    tensor = tensor.clamp(0, 255).to(torch.uint8)
    
    # 去掉批量维度，假设输入Tensor的形状为(1, 3, h, w)
    tensor = tensor.squeeze(0)
    
    # 分离Y、U、V分量
    y = tensor[0, :, :].cpu().numpy()
    u = tensor[1, :, :].cpu().numpy()
    v = tensor[2, :, :].cpu().numpy()
    
    # 根据YUV格式调整数据
    if format == 'YUV444p':
        # 对于YUV444p格式，直接按Y、U、V顺序存储每个像素的值
        with open(filename, 'wb') as f:
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())
    elif format == 'YUV420p':
        # 对于YUV420p格式，需要对U和V分量进行下采样
        u_downsampled = u[::2, ::2]
        v_downsampled = v[::2, ::2]
        with open(filename, 'wb') as f:
            f.write(y.tobytes())
            f.write(u_downsampled.tobytes())
            f.write(v_downsampled.tobytes())
    else:
        raise ValueError("Unsupported YUV format")

import torch
print(torch.backends.cudnn.version())