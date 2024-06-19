import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
import torch
import numpy as np

class CE_AV(nn.Module):
    """
    0 Anger
    1 Anticipation
    2 Disgust
    3 Fear
    4 Joy
    5 Sadness
    6 Surprise
    7 Trust
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(CE_AV, self).__init__()
        self.f0 = nn.CrossEntropyLoss()

    def forward(self, y_preds, y: Tensor):
        
        op = torch.zeros(1).to(y_preds[0].device)
        
        for y_pred in y_preds:
            out = self.f0(y_pred, y)
            op += out
        
        return op

class PCCEVE8_AV(nn.Module):
    """
    0 Anger
    1 Anticipation
    2 Disgust
    3 Fear
    4 Joy
    5 Sadness
    6 Surprise
    7 Trust
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(PCCEVE8_AV, self).__init__()
        self.POSITIVE = {1, 4, 6, 7}
        self.NEGATIVE = {0, 2, 3, 5}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_preds, y: Tensor):
        
        op = torch.zeros(1).to(y_preds[0].device)
        
        for y_pred in y_preds:
            batch_size = y_pred.size(0)
            weight = [1] * batch_size

            out = self.f0(y_pred, y)
            _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
            y_pred_label = y_pred_label.squeeze(dim=1)
            y_numpy = y.cpu().numpy()
            y_pred_label_numpy = y_pred_label.cpu().numpy()
            for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
                if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                        y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                    weight[i] += self.lambda_0
            weight_tensor = torch.from_numpy(np.array(weight)).cuda()
            out = out.mul(weight_tensor)
            out = torch.mean(out)
            op += out
        
        return op


class PCCEVE8(nn.Module):
    """
    0 Anger
    1 Anticipation
    2 Disgust
    3 Fear
    4 Joy
    5 Sadness
    6 Surprise
    7 Trust
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(PCCEVE8, self).__init__()
        self.POSITIVE = {1, 4, 6, 7}
        self.NEGATIVE = {0, 2, 3, 5}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_pred: Tensor, y: Tensor):
        batch_size = y_pred.size(0)
        weight = [1] * batch_size

        out = self.f0(y_pred, y)
        _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
        y_pred_label = y_pred_label.squeeze(dim=1)
        y_numpy = y.cpu().numpy()
        y_pred_label_numpy = y_pred_label.cpu().numpy()
        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                    y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                weight[i] += self.lambda_0
        weight_tensor = torch.from_numpy(np.array(weight)).cuda()
        out = out.mul(weight_tensor)
        out = torch.mean(out)

        return out

class PCCEEk6(nn.Module):
    """
    0 anger
    1 disgust
    2 fear
    3 joy
    4 sadness
    5 surprise
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(PCCEVE8, self).__init__()
        self.POSITIVE = {3, 5}
        self.NEGATIVE = {0, 1, 2, 4}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_pred: Tensor, y: Tensor):
        batch_size = y_pred.size(0)
        weight = [1] * batch_size

        out = self.f0(y_pred, y)
        _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
        y_pred_label = y_pred_label.squeeze(dim=1)
        y_numpy = y.cpu().numpy()
        y_pred_label_numpy = y_pred_label.cpu().numpy()
        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                    y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                weight[i] += self.lambda_0
        weight_tensor = torch.from_numpy(np.array(weight)).cuda()
        out = out.mul(weight_tensor)
        out = torch.mean(out)

        return out

class PCCE_PERR(nn.Module):

    def __init__(self, lambda_0=0):
        super(PCCE_PERR, self).__init__()
        self.label_info  = np.array([2, 1, 1, 0, 0])
        self.lambda_0 = lambda_0
        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_preds: Tensor, y: Tensor):

        op = torch.zeros(1).to(y_preds[0].device)

        for y_pred in y_preds:
            batch_size = y_pred.size(0)
            weight = [1] * batch_size

            # sentiment loss
            out = self.f0(y_pred, y) 

            _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
            y_pred_label = y_pred_label.squeeze(dim=1)
            y_numpy = y.cpu().numpy()
            y_pred_label_numpy = y_pred_label.cpu().numpy()
            
            for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
                if self.label_info[y_numpy_i] != self.label_info[y_pred_label_numpy_i]:
                    weight[i] += self.lambda_0

            weight_tensor = torch.from_numpy(np.array(weight)).cuda()
            out = out.mul(weight_tensor)
            out = torch.mean(out)
            op += out

        return op


def get_loss(opt):
    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()
    elif opt.loss_func == 'ce_av':
        return CE_AV()
    elif opt.loss_func == 'pcce_ve8':
        return PCCEVE8(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'pcce_ek6':
        return PCCEEk6(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'pcce_ve8_av':
        return PCCEVE8_AV(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'pcce_perr':
        return PCCE_PERR(lambda_0=opt.lambda_0)
    
    else:
        raise Exception
