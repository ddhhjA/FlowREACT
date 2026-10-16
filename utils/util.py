from datetime import datetime
from math import cos, pi
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import torch.nn.functional as F
from omegaconf import OmegaConf
import os
import yaml
from torch.backends import cudnn
import random
import numpy as np
import torch
import json
from framework.metrics.metric import baseline_random, baseline_mime
from framework.utils.compute_metrics import compute_metrics


def set_seed(seed: int, deterministic: bool = True) -> None:
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path=None):
    cli_conf = OmegaConf.from_cli()
    model_conf = OmegaConf.load(cli_conf.pop('config') if config_path is None else config_path)
    return OmegaConf.merge(model_conf, cli_conf)


def load_config_from_file(path):
    return OmegaConf.load(path)


def get_logging_path(log_dir):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    lod_dir = os.path.join(log_dir, time_str)
    return lod_dir


def get_tensorboard_path(tb_dir):
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = '-'.join(time_str.split(' '))
    time_str = time_str.split('.')[0]
    tb_dir = os.path.join(tb_dir, time_str)
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


def store_config(config):
    # store config to directory
    dir = config.trainer.out_dir
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img


def _fix_image(image):
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]
    return image


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_accuracy(pred, labels):
    """Computes the precision@k for the binary classification"""
    pred = (pred.cpu().detach().numpy() > 0.5).astype(int)
    labels = labels.cpu().detach().numpy()

    correct = np.sum(pred == labels)
    return correct / len(labels)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
