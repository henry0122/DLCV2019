import os
import torch

import parser
import models
import data
from mean_iou_evaluate import mean_iou_score

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

def evaluate(model, data_loader, device, args):

    ''' set model to evaluate mode '''
    model.eval()
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        iou_outputs = []
        iou_labels = []
        for idx, (imgs, lb, img_paths) in enumerate(data_loader):
            imgs = imgs.to(device)
            output, result = model(imgs)
            iou_outputs.append(result.cpu().detach().numpy())
            iou_labels.append(lb.numpy())

        iou_outputs = np.concatenate(iou_outputs)
        iou_labels = np.concatenate(iou_labels)
        
        mean_iou = mean_iou_score(iou_outputs, iou_labels)
    
    return output, mean_iou
