import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt

# === 1. 读取图像并转为 YUV ===
img = cv2.imread(r'D:\data\coco\train\000000002658.jpg')        # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# === 2. 提取色度通道（U 分量） ===
U = img_yuv[:, :, 2]  # shape: [H, W], uint8
U_normalized = U/255.0 # mark_boundaries 需要 float 格式

# 将 U 转为 float32，便于计算
U_float = U.astype(np.float32)

# 获取 1% - 99% 分位数，去除极值
low, high = np.percentile(U_float, [1, 99])

# 截断归一化
U_norm = np.clip((U_float - low) / (high - low), 0, 1)
U_norm = U_norm ** 0.8 

# === 3. 运行 SLIC（建议用亮度或原图） ===
segments = slic(img_rgb, n_segments=1000, compactness=2.5, start_label=0)

# # === 4. 在 U 分量上绘制边界 ===
# U_with_boundary = mark_boundaries(U, segments, color=(1, 0, 0))  # 红色边界

# # === 5. 可视化 ===
# plt.figure(figsize=(12, 12))
# # plt.imshow(U_with_boundary, cmap='gray')
# plt.imshow(U_normalized, cmap='gray')
# # plt.imshow(U_with_boundary)
# plt.axis('off')
# # plt.title("SLIC Boundaries on U Channel")
# plt.show()
boundary_mask = find_boundaries(segments)
U_rgb = np.stack([U_norm]*3, axis=-1) 
plt.imshow(U_rgb)
plt.axis('off')
plt.savefig('D:/colorization/CSA-LIC/superpixel_vusualization/output_original.png', bbox_inches='tight', pad_inches=0)
# U_rgb[boundary_mask] = [1, 0, 0]
U_rgb[boundary_mask] = [50/255,50/255,255/255]
plt.imshow(U_rgb)
plt.axis('off')
plt.savefig('D:/colorization/CSA-LIC/superpixel_vusualization/output.png', bbox_inches='tight', pad_inches=0)
plt.show()