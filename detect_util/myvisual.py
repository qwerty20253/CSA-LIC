import os

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# 设置配置
cfg = get_cfg()
cfg.merge_from_file("D:/code_2/CLIP-main/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # 模型配置文件路径
# cfg.MODEL.WEIGHTS = "path/to/model.pth"      # 模型权重文件路径

# 创建预测器
predictor = DefaultPredictor(cfg)

rgb_img_root = "D:/code_2/DDColor-master/color/mse/visualization"
save_dir = "D:/data/coco/visual_val/colorization"
images = os.listdir(rgb_img_root)
for img_name in images:
    # 读取图像
    # img = cv2.imread("D:/data/coco/gray/val_gray_jpg/000000000785.jpg")
    root = rgb_img_root + "/" + img_name
    save_root = save_dir + "/" + img_name
    img = cv2.imread(root)
    # 使用模型进行预测
    outputs = predictor(img)

    # 获取预测结果
    instances = outputs["instances"].to("cpu")

    # 设置置信度阈值
    conf_threshold = 0.5  # 可以根据需要调整这个阈值

    # 过滤预测结果
    keep = instances.scores > conf_threshold
    filtered_instances = instances[keep]

    # 可视化过滤后的预测结果
    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    v = v.draw_instance_predictions(filtered_instances)
    # 显示图像
    # cv2.imshow("Fast R-CNN Predictions", v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存可视化结果
    cv2.imwrite(save_root, v.get_image()[:, :, ::-1])
    print(save_root)