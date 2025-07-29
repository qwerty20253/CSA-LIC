# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import math
import os
import random
import shutil
import sys
import subprocess
import time

import cv2
import tqdm
# from pytorch_msssim import ms_ssim

from dataset import yuv_image_color_spuerpixel, yuv_image_color_val_super_pixel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from compress.datasets import ImageFolder
from compress.zoo import models
import math




class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, epoch, output, target, Y, UV):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["rec_color_mse_loss"] = self.mse(yuv_to_rgb(output["colored_img"]) * 255, yuv_to_rgb(target) * 255)
        # 计算 Detectron2 的任务损失
        # out["task_loss"] = self.task_loss_fn(yuv_to_rgb(output["colored_img"]), yuv_to_rgb(target))
        out["task_loss"] = out["rec_color_mse_loss"]

        out["loss"] = self.lmbda * (out["rec_color_mse_loss"] + out["task_loss"] / 100)/1.7 + out["bpp_loss"]
        # out["loss"] = self.lmbda * (out["rec_color_mse_loss"]) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer




import torch.nn.functional as F


def rgb_to_yuv_8bit(rgb_tensor):
    """
    将 RGB 转换为 YUV，范围限制为 0-255 并转换为 uint8。
    """
    # 确保输入为浮点数，并归一化到 0-1（假设输入是 0-255 的整数）
    # rgb_tensor = rgb_tensor.float() / 255.0

    # 提取 RGB 通道
    R, G, B = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]

    # 计算 Y, U, V
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.147 * R - 0.289 * G + 0.436 * B
    V = 0.615 * R - 0.515 * G - 0.100 * B

    # 调整 U 和 V 的范围到 [0, 255]
    Y = Y
    U = U + 0.5  # 中心点平移到 128
    V = V + 0.5  # 中心点平移到 128

    # 转为 uint8 类型
    YUV = torch.stack([Y, U, V], dim=1)
    return YUV


def yuv_to_rgb(yuv_tensor):
    """
    将 YUV444 8-bit 格式转换为 RGB。
    输入:
        yuv_tensor: 形状为 (b, 3, h, w) 的张量，其中 b 是批量大小
    输出:
        rgb_tensor: 形状为 (b, 3, h, w) 的张量，值范围为 [0, 255]，数据类型为 uint8
    """
    # 提取 Y, U, V 分量
    Y = yuv_tensor[:, 0, :, :]  # (b, h, w)
    U = yuv_tensor[:, 1, :, :] - 0.5  # 中心点平移到 [-128, 127]
    V = yuv_tensor[:, 2, :, :] - 0.5  # 中心点平移到 [-128, 127]

    # 计算 R, G, B
    R = Y + 1.140 * V
    G = Y - 0.394 * U - 0.581 * V
    B = Y + 2.032 * U

    # # 限制值范围到 [0, 255]
    # R = R.clamp(0, 255)
    # G = G.clamp(0, 255)
    # B = B.clamp(0, 255)

    # 堆叠为 RGB 张量
    rgb_tensor = torch.stack([R, G, B], dim=1)  # 转换为 uint8
    return rgb_tensor

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    enc_time = 0
    dec_time = 0
    loss = AverageMeter()

    with torch.no_grad():
        # for a in test_dataloader:
        for i, a in enumerate(tqdm.tqdm(test_dataloader)):
            img_name = a[1]
            img = a[0].to(device)
            d = rgb_to_yuv_8bit(img)
            h, w = d.size(2), d.size(3)
            p = 64
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            # d_padded = F.pad(
            #     d,
            #     (padding_left, padding_right, padding_top, padding_bottom),
            #     mode="constant",
            #     value=0,
            # )
            pad = nn.ReflectionPad2d((padding_left, padding_right, padding_top, padding_bottom))
            d_padded = pad(d)
            Y = d_padded[:, 0:1, :, :]
            UV = d_padded[:, 1:3, :, :]
            b = time.time()
            out_enc = model.compress(Y, UV)
            m = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            total_time += (e - b)
            enc_time += (m - b)
            dec_time += (e - m)
            out_dec["x_hat"] = F.pad(
                out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )
            num_pixels = d.size(0) * d.size(2) * d.size(3)
            print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
            # print(f'MS-SSIM: {compute_msssim(d, out_dec["x_hat"]):.2f}dB')
            print(f'PSNR: {compute_psnr(d, out_dec["x_hat"]):.2f}dB')
            print(f'time: {(e - b):.2f}ms')
            Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            PSNR += compute_psnr(d, out_dec["x_hat"])
            # MS_SSIM += compute_msssim(d, out_dec["x_hat"])

            rgb_img = yuv_to_rgb(out_dec["x_hat"]) * 255
            save_dir = args.save_img + "/" + img_name[0]
            tensor_cpu = rgb_img.cpu()
            array = tensor_cpu.squeeze(0).permute(1, 2, 0).numpy()
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            # 保存图像
            cv2.imwrite(save_dir, array)

        PSNR = PSNR / i
        MS_SSIM = MS_SSIM / i
        Bit_rate = Bit_rate / i
        total_time = total_time / i
        enc_time = enc_time / i
        dec_time = dec_time / i
        print(f'average_PSNR: {PSNR:.2f}dB')
        print(f'average_MS-SSIM: {MS_SSIM:.4f}')
        print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
        print(f'average_time: {total_time:.3f} ms')
        print(f'average_enc_time: {enc_time:.3f} ms')
        print(f'average_dec_time: {dec_time:.3f} ms')
    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8] + "_best" + filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="super_pixel_color_f_g",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="/data/zgy/data/coco/train2017", help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=2.5e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--txt-dir",
        type=str,
        # required=True,
        default="/data/cry/code/zgy_paper/CSA-LIC/save_img_cry/txt/5e-3",
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--save-img", type=str, 
        # required=True,
        default=rf"/data/cry/code/zgy_paper/CSA-LIC/save_img_cry",
        help="Where to Save model"
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        default=3,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=float, default=1234, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, 
                        # required=True,
                        default="/data/cry/code/zgy_paper/CSA-LIC/save_model/5e-3/model_best.pth.tar",
                        help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def count_parameters(model):
    """统计模型参数量并返回以M为单位的参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6  # 转换为百万单位

def main(argv):
    args = parse_args(argv)
    print(args)
    os.makedirs(args.txt_dir, exist_ok=True)
    os.makedirs(args.save_img, exist_ok=True)
    torch.cuda.set_device(args.use_gpu)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    test_dataset = yuv_image_color_val_super_pixel(root_dir=rf"/data/cry/code/feature_compression/datasets/COCO/val2017")
    # device = "cuda"
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model]()
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    criterion = RateDistortionLoss(lmbda=args.lmbda).cuda()

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    total_params = count_parameters(net)
    print(f"模型参数量: {total_params:.4f}M")  # 保留两位小数

    net.update()
    epoch = 0
    loss = test_epoch(epoch, test_dataloader, net, criterion, args)


if __name__ == "__main__":
    main(sys.argv[1:])