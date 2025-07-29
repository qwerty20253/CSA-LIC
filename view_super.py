import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt

# === 1. 读取图像并转为 YUV ===
img = np.load(rf"D:\colorization\CSA-LIC\demo\000000000632.npy")      # BGR


# === 2. 提取色度通道（U 分量） ===
U = img[:, :, 1]  # shape: [H, W], uint8
U_normalized = U/255.0 # mark_boundaries 需要 float 格式
U_norm = (U - U.min()) / (U.max() - U.min() + 1e-5)
U_rgb = np.stack([U_norm]*3, axis=-1) 
plt.imshow(U_rgb)
plt.axis('off')
plt.show()