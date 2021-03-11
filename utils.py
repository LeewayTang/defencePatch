import numpy as np
import cv2
import torch

def init_patch_rectangle(patch_ratio):
    # imgsz = image_len**2
    patch_width = int((299*patch_ratio*2)**0.5)
    patch_length = patch_width*20
    r = np.random.choice(2)
    if r == 0:
        patch = np.random.rand(1, 3, patch_width, patch_length)
    else:
        patch = np.random.rand(1, 3, patch_length, patch_width)
    return patch, patch.shape

def rectangle_attaching(patch, data_shape, patch_shape):
    x = np.zeros(data_shape)
    p_l, p_w = patch_shape[-1], patch_shape[-2]
    for i in range(x.shape[0]):
        random_x = np.random.choice(299-patch_shape[-1])
        random_y = np.random.choice(299-patch_shape[-2])
        for j in range(3):
            x[i][j][random_y:random_y + patch_shape[-2], random_x:random_x + patch_shape[-1]] = patch[i][j]
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    return x,mask
