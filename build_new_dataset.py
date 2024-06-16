'''
运行前的准备：
1. 在根目录下新建3个文件夹：data_train_0，data_train_1，data_train_2。
2. 在上述三个文件夹下分别都新建2个文件夹：images，masks。
'''
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from torchvision.io import read_image
import torch.nn.functional as F
import torch
import numpy as np
import os
import tifffile as tiff
import cv2
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


grid_size = 256
bcss_images_path = '/local/scratch/BCSS/BCSS_train/images'
bcss_masks_path = '/local/scratch/BCSS/BCSS_train/masks'


def load_image_as_tensor(img_path: str, is_mask: bool = True) -> torch.Tensor:
    img = tiff.imread(img_path)
    if is_mask:
        return torch.tensor(img).unsqueeze(0)
    else:
        return torch.tensor(img).permute(2, 0, 1)


def calc_padding_size(img_tensor: torch.Tensor) -> Tuple[int, int, int, int]:
    h, w = img_tensor.shape[-2:]
    pad_h_size = (h // grid_size + 1) * grid_size - h
    pad_w_size = (w // grid_size + 1) * grid_size - w
    return 0, pad_w_size, 0, pad_h_size


def cutting_image(img_tensor: torch.Tensor, img_idx: int, output: str, is_mask: bool = True) -> None:
    tensor2pil = ToPILImage()
    img_pil = tensor2pil(img_tensor)
    width, height = img_pil.size
    cols = width // grid_size
    rows = height // grid_size
    counter = 0
    for i in range(cols):
        for j in range(rows):
            box = (i * grid_size, j * grid_size, (i + 1) * grid_size, (j + 1) * grid_size)
            grid = img_pil.crop(box)
            if is_mask:
                grid.save(os.path.join(output+'/masks', f'grid_{img_idx}_{counter}_mask.gif'))
            else:
                grid.save(os.path.join(output+'/images', f'grid_{img_idx}_{counter}.png'))
            counter += 1


def build_dataset(bcss: List, is_mask: bool = True) -> None:
    for i, image in enumerate(bcss):
        img = load_image_as_tensor(img_path=image, is_mask=is_mask)
        padding_size = calc_padding_size(img_tensor=img)
        img_padded = F.pad(input=img, pad=padding_size, mode='reflect')
        if i < 10:
            cutting_image(img_tensor=img_padded, img_idx=i, output='./data_train_0', is_mask=is_mask)
        elif i >= 10 and i < 20:
            cutting_image(img_tensor=img_padded, img_idx=i, output='./data_train_1', is_mask=is_mask)
        elif i >= 20 and i < 30:
            cutting_image(img_tensor=img_padded, img_idx=i, output='./data_train_2', is_mask=is_mask)
        else:
            break


bcss_images_list = [os.path.join(bcss_images_path, file) for file in os.listdir(bcss_images_path) if os.path.isfile(os.path.join(bcss_images_path, file))]
bcss_masks_list = [os.path.join(bcss_masks_path, file) for file in os.listdir(bcss_masks_path) if os.path.isfile(os.path.join(bcss_masks_path, file))]
build_dataset(bcss=bcss_images_list, is_mask=False)
build_dataset(bcss=bcss_masks_list, is_mask=True)
