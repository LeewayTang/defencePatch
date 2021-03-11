import pretrainedmodels
import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
from utils import *
import pretrainedmodels.utils as putils
import torch.nn.functional as F
from PIL import ImageFilter
from os import listdir
import warnings

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--manualSeed', type=int, default=138)
    parser.add_argument('--netClassifier', default='inceptionv3')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--patch_ratio', type=int, default=0.3)
    parser.add_argument('--target_class', type=int, default=919)
    parser.add_argument('--target_conf', type=int, default=0.9)
    parser.add_argument('--source', type=str, default='./test_imgs/')
    parser.add_argument('--max_count', type=int, default=50)
    return parser.parse_args()

def just_do_it(opt):

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

    x = netClassifier.input_size
    x.insert(0, 1)
    data_shape = tuple(x)
    patch0, patch_shape = init_patch_rectangle(opt.patch_ratio)

    has_trained = False

    for filename in listdir(opt.source):
        img_dir = opt.source + filename
        if img_dir.lower().endswith('.jpg'):
            input_tensor = load_img2tensor(img_dir, netClassifier)

            patch1, mask1 = rectangle_attaching(patch0, data_shape, patch_shape)
            patch, mask = torch.Tensor(patch1), torch.Tensor(mask1)
            adv_tensor = torch.mul((1-mask), input_tensor) + torch.mul(mask, patch)
            adv_tensor.requires_grad_(True)
            adv_out = F.softmax(netClassifier(adv_tensor))
            target_prob = adv_out.data[0][opt.target_class]

            count = 0

            while target_prob < opt.target_conf:
                count += 1

                loss = -torch.log(F.softmax(adv_out)[0][opt.target_class])
                loss.backward()

                adv_grad = adv_tensor.grad.clone()
                adv_tensor.grad.data.zero_()

                patch -= adv_grad

                adv_tensor = torch.mul((1-mask), input_tensor) + torch.mul(mask, patch)
                adv_tensor.requires_grad_(True)

                adv_out = F.softmax(netClassifier(adv_tensor))
                target_prob = adv_out.data[0][opt.target_class]

                print(count)
                if count >= opt.max_count:
                    break

            has_trained = True

    if has_trained:
        save_image(patch, './patch.jpg', normalize=True)
        save_image(mask, './mask.jpg', normalize=True)
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    opt = get_args()
    just_do_it(opt)