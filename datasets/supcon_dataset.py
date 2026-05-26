import os
import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
from ylib.scipy_misc import imread, imsave
from .meta import DEVICE_INFOS
from torchvision import transforms
import io
import torchvision.transforms as T

# 设置随机种子，以确保结果可重复
torch.manual_seed(0)
#torch.cuda.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)

def crop_face_from_scene(image, bbox, scale):
    """
    从给定图像中裁剪人脸区域。
    参数:
    - image: 输入图像。
    - bbox: 人脸区域的边界框（x1, y1, x2, y2）。
    - scale: 缩放比例。
    返回:
    - 裁剪后的图像区域。
    """
    x1, y1, x2, y2 = [float(ele) for ele in bbox]
    h = y2 - y1
    w = x2 - x1
    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - h_scale / 2.0
    x1 = x_mid - w_scale / 2.0
    y2 = y_mid + h_scale / 2.0
    x2 = x_mid + w_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), h_img)
    x2 = min(math.floor(x2), w_img)
    region = image[y1:y2, x1:x2]
    return region

class FaceDataset(Dataset):
    """
    人脸数据集类，继承自 PyTorch 的 Dataset，用于加载和处理人脸反欺骗检测数据集。
    """
    def __init__(self,
                 dataset_name,
                 root_dir,
                 split='train',
                 label=None,
                 transform=None,
                 scale_up=1.1,
                 scale_down=1.0,
                 map_size=32,
                 UUID=-1):
        # 初始化数据集参数
        if self.split = 'train':  # 数据集的分割（train、dev、test）
            splits = ["train", "dev"]
        else:
            splits = ['test']
        #self.video_list = os.listdir(root_dir)  # 数据集目录下的视频列表
        # 收集对应视频文件夹路径
        self.video_list = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
            and any(s in d.lower() for s in splits)
        ]
        if label is not None and label != 'all':
            self.video_list = list(filter(lambda x: label in x, self.video_list))  # 根据标签过滤视频列表
        self.dataset_name = dataset_name  # 数据集名称
        self.root_dir = root_dir  # 数据集根目录
        self.transform = transform  # 图像变换
        self.scale_up = scale_up  # 缩放参数
        self.scale_down = scale_down  # 缩放参数
        self.map_size = map_size  # 映射大小
        self.UUID = UUID  # 唯一标识符
        self.face_width = 400  # 人脸宽度

    def __len__(self):
        # 返回数据集中视频的数量
        return len(self.video_list)

    def get_client_from_video_name(self, video_name):
        """
        根据视频名称提取客户端 ID。
        """
        if 'msu' in self.dataset_name.lower() or 'replay' in self.dataset_name.lower():
            match = re.findall('client(\d\d\d)', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'oulu' in self.dataset_name.lower():
            match = re.findall('(\d+)_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'casia' in self.dataset_name.lower():
            match = re.findall('_(\d+)_[H|N][R|M]_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'celeba' in self.dataset_name.lower():
            match = re.findall('_(\d+)$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        else:
            raise RuntimeError("no dataset found")
        return client_id

    def __getitem__(self, idx):
        """
        获取数据集中指定索引处的视频样本。
        """
        image_dir = self.video_list[idx]
        video_name = os.path.basename(image_dir)

        #video_name = self.video_list[idx]  # 视频名称
        spoofing_label = int('live' in video_name)  # 判断视频是否为活体
        if self.dataset_name in DEVICE_INFOS:
            # 根据设备信息匹配视频标签
            if 'live' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['live']
            elif 'spoof' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['spoof']
            else:
                raise RuntimeError("STH WRONG")
            device_tag = None
            for pattern in patterns:
                len1 = len(re.findall(pattern, video_name))
                #print(len1)
                if len1 > 0:
                    if device_tag is not None:
                        raise RuntimeError("Multiple Match")
                    device_tag = pattern
            if device_tag is None:
                raise RuntimeError("No Match")
        else:
            device_tag = 'live' if spoofing_label else 'spoof'

        client_id = self.get_client_from_video_name(video_name)  # 获取客户端 ID

        image_dir = os.path.join(self.root_dir, video_name)  # 构建图像目录路径
        # 过滤掉 None
        #self.transform = transforms.Compose([t for t in self.transform.transforms if t is not None])
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需要调整尺寸
            transforms.ToTensor(),
            normalizer
        ])
        
        if self.split == 'train':
            # 如果是训练集，进行图像采样和数据增强
            image_x, info, _ = self.sample_image(image_dir)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            # 如果是验证或测试集，只进行图像采样
            image_x, info, _ = self.sample_image(image_dir)
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = image_x_view1

        # 构建返回样本，包括图像、标签、设备标签、视频名称、客户端 ID 和关键点信息
        sample = {
            "image_x_o": np.array(transform(PIL.Image.fromarray(image_x))),
            "image_x_v1": np.array(image_x_view1),
            "image_x_v2": np.array(image_x_view2),
            "label": spoofing_label,
            "UUID": self.UUID,
            'device_tag': device_tag,
            'video': video_name,
            'client_id': client_id,
            'points': info['points']
        }
        return sample

    def sample_image(self, image_dir):
        """
        从视频目录中随机采样一帧图像。
        """
        frames = glob(os.path.join(image_dir, "org_*.jpg"))  # 获取所有原始帧图像
        frames_total = len(frames)  # 总帧数
        if frames_total == 0:
            raise RuntimeError(f"{image_dir}")

        for temp in range(500):
            if temp > 200:
                image_id = int(re.findall('_(\d+).jpg', frames[0])[0]) // 5  # 备份策略
            else:
                image_id = np.random.randint(0, frames_total)  # 随机选择图像 ID

            image_name = f"crop_{image_id*5:04d}.jpg"  # 构建裁剪图像名称
            info_name = f"infov1_{image_id*5:04d}.npy"  # 构建信息文件名称

            image_path = os.path.join(image_dir, image_name)  # 图像路径
            info_path = os.path.join(image_dir, info_name)  # 信息路径

            if os.path.exists(image_path) and os.path.exists(info_path):
                break

        info = np.load(info_path, allow_pickle=True).item()  # 加载信息文件
        image = imread(image_path)  # 读取图像

        return image, info, image_id * 5

    def generate_square_images(self, image, info, range_scale=3):
        """
        生成正方形图像。
        """
        points = np.array(info['points'])
        dist = lambda p1, p2: int(np.sqrt(((p1 - p2) ** 2).sum()))
        width = dist(points[0], points[1])  # 计算宽度
        center = tuple(points[2])  # 中心点

        angle = math.degrees(math.atan((points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])))  # 计算旋转角度
        rect = (center, (int(width * range_scale), int(width * range_scale)), angle)
        img_rows = image.shape[0]
        img_cols = image.shape[1]

        round = 0
        initial_scale = range_scale
        scale = range_scale
        min_scale = (256 / self.face_width) * initial_scale + 0.3

        while True:
            if inside_rect(rect=rect, num_cols=img_cols, num_rows=img_rows):
                break

            if scale < min_scale:
                pad_size = 300
                image = np.array(tf.pad(PIL.Image.fromarray(image), pad_size, padding_mode='symmetric'))
                center = (center[0] + pad_size, center[1] + pad_size)
                rect = (center, (int(width * scale), int(width * scale)), angle)
                break

            scale = range_scale - round * 0.1
            rect = (center, (int(width * scale), int(width * scale)), angle)
            round += 1

        scaled_face_size = int(self.face_width * scale / initial_scale)
        image_square_cropped = crop_rotated_rectangle(image=image, rect=rect)
        image_resized = cv2.resize(image_square_cropped, (scaled_face_size, scaled_face_size))
        return image_resized

    def get_single_image_x(self, image_dir):
        """
        获取单个图像样本并调整大小。
        """
        image, info = self.sample_image(image_dir)
        h_img, w_img = image.shape[0], image.shape[1]
        h_div_w = h_img / w_img
        image_x = cv2.resize(image, (self.face_width, int(h_div_w * self.face_width)))
        return image_x


class Identity():  # used for skipping transforms
    def __call__(self, im):
        return im


class RandomCutout(object):
    def __init__(self, n_holes, p=0.5):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            p (int): probability to apply cutout
        """
        self.n_holes = n_holes
        self.p = p

    def rand_bbox(self, W, H, lam):
        """
        Return a random box
        """
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.rand(1) > self.p:
            return img

        h = img.size(1)
        w = img.size(2)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
        for n in range(self.n_holes):
            img[:, bby1:bby2, bbx1:bbx2] = img[:, bby1:bby2, bbx1:bbx2].mean(dim=[-2, -1], keepdim=True)
        return img
    
    
class JPEGCompression:
    def __init__(self, quality_range=(30, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = PIL.Image.open(buffer)
        return compressed_img
    
class AddCameraSensorNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        img: PIL.Image or Tensor (C x H x W)
        """
        # 如果是 PIL 图像，先转成 tensor
        if not isinstance(img, torch.Tensor):
            img = tf.to_tensor(img)  # 转为 float tensor, range [0, 1]

        noise = torch.randn_like(img) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        return tf.to_pil_image(noisy_img)