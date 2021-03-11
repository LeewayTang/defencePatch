import numpy as np
import torch.backends.cudnn as cudnn
import torch
import pretrainedmodels
import torchvision.transforms as transforms
import pretrainedmodels.utils as putils
import torch.nn.functional as F

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

def load_img2tensor(img_dir, netClassifier):
    load_img = putils.LoadImage()
    tf_img = putils.TransformImage(netClassifier)

    input_img = load_img(img_dir)
    input_tensor0 = tf_img(input_img)
    input_tensor = input_tensor0.unsqueeze(0)

    return input_tensor

def just_classify(opt):
    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
    if opt.cuda:
        netClassifier.cuda()
    normalize = transforms.Normalize(mean=netClassifier.mean, std=netClassifier.std)
    netClassifier.eval()
    load_img = putils.LoadImage()
    tf_img = putils.TransformImage(netClassifier)
    path_img = "stop.jpeg"
    input_img = load_img(path_img)
    # input_img = input_img.filter(ImageFilter.GaussianBlur(radius=2))
    input_tensor0 = tf_img(input_img)
    input_tensor = input_tensor0.unsqueeze(0)
    input_tensor.requires_grad_(True)
    prediction = netClassifier(input_tensor)

    return F.softmax(prediction)[0][opt.target_class]