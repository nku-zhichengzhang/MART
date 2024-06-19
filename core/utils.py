import os
import datetime
import shutil
import math

import random
from transforms.spatial import Preprocessing
from sklearn import metrics
from dataset_info import label_info
import numpy as np
import torch
from sklearn.metrics import classification_report
from copy import deepcopy

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def setup_seed(seed=3407):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def local2global_path(opt):
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.audio_path = os.path.join(opt.root_path, opt.audio_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.srt_path = os.path.join(opt.root_path, opt.srt_path)
        if opt.debug:
            opt.result_path = "debug"
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.expr_name == '':
            now = datetime.datetime.now()
            now = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            os.makedirs(opt.result_path)

        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exception

def get_spatial_transform(opt, mode):
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=True, center=False)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
    else:
        raise Exception


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def process_data_item(opt, data_item):
    visual, target, audio, visualization_item,_ = data_item
    target = target.cuda()

    visual = visual.cuda()
    audio = audio.cuda()
    assert visual.size(0) == audio.size(0)
    batch = visual.size(0)
    return visual, target, audio, visualization_item, batch

def process_data_item_w_language(opt, data_item):
    visual, target, audio, visualization_item, words = data_item
    text = []
    for vid in range(visual.size(0)):
        vid_text = []
        for seg in range(visual.size(1)):
            vid_text.append(words[seg][vid])
        text.append(vid_text)
    target = target.cuda()
    # words = words.cuda()
    visual = visual.cuda()
    audio = audio.cuda()
    assert visual.size(0) == audio.size(0)
    batch = visual.size(0)
    return visual, target, audio, visualization_item, text, batch

def run_model(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False, isTrain=True):
    if opt.alg=='VAANet':
        visual, target, audio = inputs
        outputs = model(visual, audio)
        y_pred, alpha, beta, gamma, temporal_score = outputs
        loss = criterion(y_pred, target)
    else:
        visual, target, audio = inputs
        outputs = model(visual, audio)
        y_pred, temporal_score = outputs
        loss = criterion(y_pred, target)
    return y_pred, loss, temporal_score

def run_model_language(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False, isTrain=True):
    if opt.alg=='VAANet':
        visual, target, audio, words = inputs
        outputs = model(visual, audio, words)
        y_pred, alpha, beta, gamma, temporal_score = outputs
        loss = criterion(y_pred, target)
    else:
        visual, target, audio, words = inputs
        outputs = model(visual, audio, words)
        y_pred, temporal_score = outputs
        loss = criterion(y_pred, target)
    return y_pred, loss, temporal_score


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size
